
# about

This is just a small pet project, the idea is to create a treat dispenser that is triggered by a button push (from a dog), then indicate one of a few simple commands (for example: sit, down, up), take a picture of the dog and if the dog performed the trick, they get a treat

# setup

WARN: This requires a local installation of the modules because of the dependencies between the packages so it's better to use a virtual environment for this

So I am pretty much putting all the code in this same repository, so you can find all dependencies in the `requirements.txt` file so you can install it such as:

```
pip install -r requirements.txt
```

Additinally, you need to run a local installation so that the sibiling packages can see each other, so you need tu run:

```
pip install -e .
```

# what you'll find here

## localml

It's the project that contains all ML related things: models, training, testing, etc.

For the `train_model.py` to run, you need to create an `images` folder inside the `localml` module and place your image dataset similar to this:

```
images
├── test
│   ├── standingdog
│   │   ├── yourimages.jpg
│   └── sittingdog
│       ├── yourimages.jpg
└── train
    ├── standingdog
    │   ├── yourimages.jpg
    └── sittingdog
        ├── yourimages.jpg
```

NOTICE: I pretty much hardcoded a few things to work with JPG images, so you might want to use JPG as well

## webapi

This is a flask project to expose a very simple API that pretty much takes the model from the output of the training and exposes a set of endpoints to authenticate and perform a prediction for an input image