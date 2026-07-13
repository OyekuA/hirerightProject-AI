import abc
from typing import Optional
import httpx
import structlog

logger = structlog.get_logger()


class MeetingBotClient(abc.ABC):
    @abc.abstractmethod
    async def inject_bot(
        self,
        meeting_url: str,
        session_id: str,
        candidate_id: int,
        join_at: Optional[str] = None,
        bot_name: Optional[str] = None,
    ) -> str:
        """Inject a bot into the meeting and return the bot ID."""

    @abc.abstractmethod
    async def create_transcript(self, recording_id: str) -> str:
        """Request async transcript creation and return a transcript job ID."""

    @abc.abstractmethod
    async def fetch_transcript(self, transcript_id: str) -> list[dict]:
        """Fetch completed transcript turns. Returns a list of turn dicts."""


class RecallAIClient(MeetingBotClient):
    _REGION_HOSTS: dict[str, str] = {
        "us-east-1": "us-east-1.recall.ai",
        "us-west-2": "us-west-2.recall.ai",
        "eu-central-1": "eu-central-1.recall.ai",
        "ap-northeast-1": "ap-northeast-1.recall.ai",
    }

    def __init__(
        self,
        api_key: str,
        region: str = "us-east-1",
        base_url: str | None = None,
    ):
        self.api_key = api_key
        self.region = region
        if base_url is not None:
            self._base_url = base_url
        else:
            host = self._REGION_HOSTS.get(region)
            if host is None:
                raise ValueError(f"Unsupported Recall.ai region: {region}")
            self._base_url = f"https://{host}/api/v1"

    async def _post(self, path: str, json_body: dict) -> dict:
        url = f"{self._base_url}{path}"
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                json=json_body,
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Accept": "application/json",
                },
                timeout=30,
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                body = exc.response.text if exc.response else ""
                logger.error("Recall.ai API error", status=exc.response.status_code, body=body[:500])
                raise RuntimeError(f"Recall.ai {exc.response.status_code}: {body[:300]}") from exc
            return resp.json()

    async def _get(self, path: str) -> dict:
        url = f"{self._base_url}{path}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                url,
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Accept": "application/json",
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()

    async def inject_bot(
        self,
        meeting_url: str,
        session_id: str,
        candidate_id: int,
        join_at: Optional[str] = None,
        bot_name: Optional[str] = None,
    ) -> str:
        body = {
            "meeting_url": meeting_url,
            "recording_config": {
                "video_mixed_mp4": {},
            },
            "metadata": {
                "session_id": session_id,
                "candidate_id": str(candidate_id),
            },
        }
        if join_at:
            body["join_at"] = join_at
        if bot_name:
            body["bot_name"] = bot_name
        body["chat"] = {
            "on_bot_join": {
                "send_to": "everyone",
                "message": "This meeting is being recorded for interview evaluation purposes.",
                "pin": True,
            },
            "on_participant_join": {
                "message": "This meeting is being recorded for interview evaluation purposes.",
                "exclude_host": False,
            },
        }

        result = await self._post("/bot/", body)
        bot_id = result.get("id")
        if not bot_id:
            raise RuntimeError(f"Recall.ai bot creation returned no id: {result}")
        logger.info("Recall.ai bot injected", bot_id=bot_id, session_id=session_id)
        return bot_id

    async def create_transcript(self, recording_id: str) -> str:
        body = {
            "provider": {"recallai_async": {"language_code": "auto"}},
            "diarization": {"use_separate_streams_when_available": True},
        }
        result = await self._post(f"/recording/{recording_id}/create_transcript/", body)
        transcript_id = result.get("id")
        if not transcript_id:
            raise RuntimeError(f"Recall.ai transcript creation returned no id: {result}")
        logger.info("Recall.ai transcript created", transcript_id=transcript_id, recording_id=recording_id)
        return transcript_id

    async def fetch_transcript(self, transcript_id: str) -> list[dict]:
        result = await self._get(f"/transcript/{transcript_id}/")
        data = result.get("data", {}) if isinstance(result, dict) else {}
        download_url = data.get("download_url") if isinstance(data, dict) else None
        if not download_url:
            raise RuntimeError(f"Recall.ai transcript {transcript_id} has no data.download_url")
        async with httpx.AsyncClient() as client:
            resp = await client.get(download_url, timeout=30)
            resp.raise_for_status()
            transcript_data = resp.json()
        turns = transcript_data if isinstance(transcript_data, list) else transcript_data.get("turns", [])
        logger.info("Recall.ai transcript fetched", transcript_id=transcript_id, turns=len(turns))
        return turns


class MeetingBaaSClient(MeetingBotClient):
    async def inject_bot(
        self,
        meeting_url: str,
        session_id: str,
        candidate_id: int,
        join_at: Optional[str] = None,
        bot_name: Optional[str] = None,
    ) -> str:
        raise NotImplementedError("MeetingBaaSClient.inject_bot is not implemented")

    async def create_transcript(self, recording_id: str) -> str:
        raise NotImplementedError("MeetingBaaSClient.create_transcript is not implemented")

    async def fetch_transcript(self, transcript_id: str) -> list[dict]:
        raise NotImplementedError("MeetingBaaSClient.fetch_transcript is not implemented")
