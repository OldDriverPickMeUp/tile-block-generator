# Tile Block Generator

This repo implement a simple tile block generator like the so called wave collapse function approach.

It main based on this [dissertation](http://www.logarithmic.net/pfh/thesis).

## get started

### requiements

You will need at least python version 3. And then install all requirements

`pip install -r requirements.txt`

### folders

- /dev for development and test use
- /examples example tiles downloaded from unity store
- /generator all generator code

### scripts

- `python dev/try_output.py` It will show you the current input which is a tile block layout that the generator will refer to.
- `python dev/try_image_reader.py` This command will show you all the tiles are in use and their category number.
- `python dev/try_generator.py` This command will generate a new tile block layout base on the input/reference and save the image in current folder.
- `pythin dev/try_overlap.py` This command will generate a new tile block layout will overlap method.

## UI

I developed an browser interface to edit the example/input/reference data. In this repo: [tile-block-generator-ui](https://github.com/OldDriverPickMeUp/tile-block-generator-ui).

So it's possible to create arbitrary tile block layout reference with arbitrary tiles.

