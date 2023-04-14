# pylint: disable=E0401

import asyncio
import base_download_from_local_service as loc_srv

STAGE = "download-pollutants"


# noinspection DuplicatedCode
async def __download_current_year(args: object) -> None:
    url, request_params = loc_srv.get_params(args=args)
    _ = await loc_srv.get_response(url, request_params)


if __name__ == '__main__':
    stage_args = loc_srv.parse_args(STAGE)
    asyncio.run(__download_current_year(args=stage_args))
