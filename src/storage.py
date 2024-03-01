"""https://elasticsearch-py.readthedocs.io/en/latest/async.html"""
from __future__ import annotations

import os
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from src.config import (logger,
                        PROJECT_ROOT_DIR)
from src.utils import jaccard_similarity
from src.data_types import TextsDeleteSample
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Base settings object to inherit from."""

    class Config:
        env_file = os.path.join(PROJECT_ROOT_DIR, ".env")
        env_file_encoding = "utf-8"


class ElasticClient(AsyncElasticsearch):
    """Handling with AsyncElasticsearch"""
    def __init__(self, *args, **kwargs):
        # self.settings = ElasticSettings()
        super().__init__(
            hosts="http://srv01.nlp.dev.msk2.sl.amedia.tech:9200",
            basic_auth=("elastic", "changeme"),
            request_timeout=100,
            max_retries=50,
            # max_hits=500,
            # chunk_size=20,
            retry_on_timeout=True,
            *args,
            **kwargs,
        )

    async def search_by_query(self, index: str, query: {}):
        """
        :param query:
        :return:
        """
        resp = await self.search(
            allow_partial_search_results=True,
            min_score=0,
            index=index,
            query=query,
            size=50)
        await self.close()
        return resp

    async def delete_by_ids(self, index_name: str, del_ids: list):
        """
        :param index_name:
        :param del_ids:
        """
        _gen = ({"_op_type": "delete", "_index": index_name, "_id": i} for i in del_ids)
        await async_bulk(
            self,
            _gen,
            # chunk_size=self.chunk_size,
            chunk_size=500,
            raise_on_error=False,
            stats_only=True,
        )