import asyncio
import download_from_local_service as loc_srv


async def download_current_year(args, params_section: str):
    url, request_params = loc_srv.get_params(
        args=args, params_section=params_section, sub_type="current_year")
    _ = await loc_srv.get_response(url, request_params)


if __name__ == '__main__':
    section = "download-weather"
    stage_args = loc_srv.parse_args(section)
    asyncio.run(download_current_year(args=stage_args, params_section=section))
