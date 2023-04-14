# pylint: disable=E0401

from argparse import ArgumentParser
import yaml
import requests


def parse_args(args_parser_name: str):
    """
    Parses command line args
    @param args_parser_name: Parser name
    @return: Parsed arguments
    """
    parser = ArgumentParser(args_parser_name)
    parser.add_argument('--service_url', required=True, help='Loader service url')
    parser.add_argument('--params', required=True, help='Path to params')
    parser.add_argument('--params_section', required=True, help='Section with params')
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


def get_params(args):
    with open(args.params, 'r', encoding='UTF-8') as file_stream:
        params = yaml.safe_load(file_stream)[args.params_section]
    url = f'{args.service_url}/{params[f"end_point"]}'
    request_params = params[f"request_params"]
    return url, request_params
