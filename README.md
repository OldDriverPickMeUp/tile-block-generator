# Tile Block Generator

This repo implements a simple tile block generator like the so called wave collapse function approach.

It's mainly based on this [dissertation](http://graphics.stanford.edu/~pmerrell/thesis.pdf).

Input Image:

<img src="https://github.com/OldDriverPickMeUp/tile-block-generator/blob/master/examples/example-json.png?raw=true" alt="input" width="300" height="300" />

Generated Image:

<img src="https://github.com/OldDriverPickMeUp/tile-block-generator/blob/master/examples/output-(15,%2015)-71482345.png?raw=true" alt="output" width="300" height="300" />


## Get Started

You can use this [tool](https://tile-block-generator-ui.sssssinter.now.sh/) to make your own example.
 
You can also go to [tile-block-generator-ui](https://github.com/OldDriverPickMeUp/tile-block-generator-ui) and build by yourself.

### Requiements

You will need at least python version 3. And then install all the requirements `pip install -r requirements.txt`.

### Generate Tile Blocks

For example
 
```python tool.py --example ./examples/example.json --folder ./examples/tiles --size 15 15```

This command generate an output tile blocks from an example file generate by [tile-block-generator-ui](https://github.com/OldDriverPickMeUp/tile-block-generator-ui).

The output layout has a size of 15 * 15. And all the tile images under folder `./examples/tiles`

Check `./examples/example-json.png` to see the image of example.

### Usage

`python tool.py --help` and you will get the following response:

```
Usage: tool.py [OPTIONS]

Options:
  --example TEXT                  example json file  [required]
  --size INTEGER...               output size  [required]
  --folder TEXT                   tile image folder  [required]
  --overlap                       enable overlap method
  --overlap-units INTEGER         overlap units
  --overlap-block-size INTEGER...
                                  overlap block size
  --output-prefix TEXT            prefix for output file
  --help                          Show this message and exit.
```


### Folders

- /dev for development and test use
- /examples/tiles/ example tiles downloaded from unity store
- /examples/example.json example reference
- /generator all generator code



