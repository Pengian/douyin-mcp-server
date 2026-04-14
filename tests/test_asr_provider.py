import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from douyin_mcp_server import server


class DummyCtx:
    def info(self, _msg: str) -> None:
        return None

    def error(self, _msg: str) -> None:
        return None

    async def report_progress(self, _done: int, _total: int) -> None:
        return None


class TestAsrProviderBehavior(unittest.IsolatedAsyncioTestCase):
    async def test_local_provider_without_dashscope_key(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DASHSCOPE_API_KEY", None)
            os.environ.pop("API_KEY", None)

            with patch.object(server.DouyinProcessor, "parse_share_url", return_value={"url": "https://example.com/v.mp4", "title": "t", "video_id": "1"}), \
                patch.object(server.DouyinProcessor, "download_video", autospec=True) as mock_download, \
                patch.object(server.DouyinProcessor, "extract_audio", autospec=True) as mock_extract_audio, \
                patch.object(server.DouyinProcessor, "extract_text_from_local_audio", return_value="local-ok"), \
                patch.object(server.DouyinProcessor, "cleanup_files", autospec=True) as mock_cleanup:
                temp_video = Path(tempfile.gettempdir()) / "douyin-test.mp4"
                temp_audio = Path(tempfile.gettempdir()) / "douyin-test.mp3"
                mock_download.return_value = temp_video
                mock_extract_audio.return_value = temp_audio

                text = await server.extract_douyin_text(
                    share_link="https://v.douyin.com/test/",
                    asr_provider="local",
                    ctx=DummyCtx(),
                )
                self.assertEqual(text, "local-ok")
                mock_cleanup.assert_called_once()

    async def test_dashscope_provider_requires_key_before_network(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DASHSCOPE_API_KEY", None)
            os.environ.pop("API_KEY", None)

            with patch.object(server.DouyinProcessor, "parse_share_url", side_effect=AssertionError("should not call network parse")):
                with self.assertRaises(Exception) as err:
                    await server.extract_douyin_text(
                        share_link="https://v.douyin.com/test/",
                        asr_provider="dashscope",
                        ctx=DummyCtx(),
                    )
                self.assertIn("DASHSCOPE_API_KEY", str(err.exception))

    async def test_invalid_provider(self) -> None:
        with self.assertRaises(Exception) as err:
            await server.extract_douyin_text(
                share_link="https://v.douyin.com/test/",
                asr_provider="invalid",
                ctx=DummyCtx(),
            )
        self.assertIn("不支持的 asr_provider", str(err.exception))


if __name__ == "__main__":
    unittest.main()
