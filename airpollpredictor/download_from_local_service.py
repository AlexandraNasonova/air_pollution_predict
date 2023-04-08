import yaml
import requests
from argparse import ArgumentParser


def parse_args(args_parser_name: str):
    parser = ArgumentParser(args_parser_name)
    parser.add_argument('--service_url', required=True, help='Loader service url')
    parser.add_argument('--params', required=True, help='Path to params')
    return parser.parse_args()


async def __post(url: str, params: dict):
    print(f"POST_REQUEST_PARAMS. url: {url}, load_params: {params['load_params']}")
    return requests.post(
        url=url,
        json=params['load_params']
    )


async def get_response(url: str, response_params: dict) -> None:
    response = await __post(url=url, params=response_params)
    if response.status_code != 200:
        print(response.url)
        print(response)
        raise ConnectionError()


def get_params(args, params_section: str, sub_type: str):
    with open(args.params, 'r') as fp:
        params = yaml.safe_load(fp)[params_section]
    url = f'{args.service_url}/{params[f"end_point_{sub_type}"]}'
    request_params = params[f"request_params_{sub_type}"]
    return url, request_params
