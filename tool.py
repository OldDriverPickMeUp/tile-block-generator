import json
import os

import click

from generator.adapter import Adapter


def validate_folder(ctx, param, value):
    if os.path.exists(value):
        return value
    else:
        raise click.BadParameter(f'folder {value} does not exist')


def validate_example(ctx, param, value):
    if os.path.exists(value):
        return value
    else:
        raise click.BadParameter(f'example {value} does not exist')


def validate_size(ctx, param, value):
    h, w = value
    if h >= 5 and w >= 5:
        return value
    else:
        raise click.BadParameter(f'height and width of size must not below 5 ')


def validate_overlap(overlap_block_size, output_size, overlap_units):
    if len(overlap_block_size) != 2:
        raise Exception("overlap block size needed")
    oh, ow = overlap_block_size
    h, w = output_size
    if oh >= h or ow >= w:
        raise Exception("overlap size should be smaller than output size")
    if overlap_units >= oh or overlap_units >= ow:
        raise Exception("overlap units should be smaller than overlap block size")


@click.command()
@click.option("--example", type=str, help="example json file", required=True, callback=validate_example)
@click.option("--size", nargs=2, type=int, help="output size", required=True, callback=validate_size)
@click.option("--folder", type=str, help="tile image folder", required=True, callback=validate_folder)
@click.option("--overlap", is_flag=True, help="enable overlap method")
@click.option("--overlap-units", type=int, help="overlap units")
@click.option("--overlap-block-size", nargs=2, type=int, help="overlap block size")
@click.option("--output-prefix", type=str, default="output", help="prefix for output file")
def generate(example, folder, size, overlap, overlap_units, overlap_block_size, output_prefix):
    with open(example) as f:
        example_data = json.load(f)
    adapter = Adapter(folder, example_data)
    if not overlap:
        return adapter.generate(size, output_prefix)
    validate_overlap(overlap_block_size, size, overlap_units)
    return adapter.generate_overlap(size, overlap_units, overlap_block_size, output_prefix)


if __name__ == '__main__':
    generate()
