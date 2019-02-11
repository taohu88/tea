# Release plan

# Version 0.0.1 (Done: 2/1/2019)
* Clean up for mnist training
* Add poor man lr finder capability
* Parse model from config (initial version)
* Using [Kaiming initialization](https://arxiv.org/abs/1502.01852)

# Version 0.0.2 (Done: 2/2/2019)
* Clean up model file parse (input/output/mutliple output)
* Initial workable solution for tinyimage data set and vgg
* Using [AdamW](https://arxiv.org/pdf/1711.05101.pdf)

# Version 0.0.3 (Done: 2/9/2019)
* Refactor app config and model config parser
* Add enum support for all app config and model config string
* Callback support in trainer
* Callback for scheduler
* More schedulers
* Callback for major metrics
* Nice print out(metrics, lr, etc)

# Version 0.0.4 Text (Done 2/9/2019)
* Support text dataset
* Train at least one text model (RNN)

# Version 0.0.5 Simple reinforcement learning (Done 2/11/2019)
* Simple reinforcement learning actor-critic without replay memory

# Version 0.0.6 (Yolo3 detect)
* Yolo V3 detect workable
* Yolo model refactor
* Easy model with most single string for cnn 

# Version 0.0.7 (Transfer)
* Transfer learning

# Version 0.0.8 (Gan)
* Add GAN example

# Version 0.0.9 Text (???)
* Support state of art awd-lstm-lm

# Version 0.1.0 (First stable code)
* Add test cases placeholders
* Add one test case for each major package
* Ready to test imagenet data set with different models

# Version 0.1.5 Dataset support such as tabular data/pandas data
* Support model build beyond text/vision

# Version 0.2.0 (Face recognition)

