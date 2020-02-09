## TTC Subway and SRT Train Service Delays 2016 - 2019

The project explores TTC Subway and SRT Train Service Delay Data to forecast duration of service delays by category in relation to weekday, time, TTC line, and other data.

The data source is City of Toronto Open Data portal [https://open.toronto.ca/dataset/ttc-subway-delay-data/](https://open.toronto.ca/dataset/ttc-subway-delay-data/).

A number of project settings could be selected: the minimum delay time, a low boundary of time, and a start date. Also, there are two options are available for defining of ranges of time delays: two or three ranges could be set.
Selection of project setting including time periods could be changed with some light rework of the script. It could be done by defining variables and uncommenting the relevant line of code.


###	Prediction of long TTC Subway service delays of 60 minutes and more

The model is built based on following settings. The minimum delay time which is considered in the project is 15 minutes. Only delays after 6:00 AM form '2016-10-01' to '2019-10-31' are evaluated. Also, two ranges of time delays are defined: 15-60 minutes and 60 minutes or more. They were selected based on personal schedule requirements.

To create and save a model run [‘ttc_2_ranges_preprocessing_and_model_creation.bat’]() file.

The model will be saved as the [‘ttc_2_delays_model.pickle’]() file in the [‘4. Insights\Models’]() folder.
