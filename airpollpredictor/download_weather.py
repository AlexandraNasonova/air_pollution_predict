# pylint: disable=E0401

import asyncio
import download_from_local_service as loc_srv

SECTION = "download-weather"


# noinspection DuplicatedCode
async def download_current_year(args: object, params_section: str) -> None:
    url, request_params = loc_srv.get_params(
        args=args, params_section=params_section, sub_type="current_year")
    _ = await loc_srv.get_response(url, request_params)


if __name__ == '__main__':
    stage_args = loc_srv.parse_args(SECTION)
    asyncio.run(download_current_year(args=stage_args, params_section=SECTION))
