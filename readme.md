# Ludwig Examples
## Deep learning made easy

Ludwig is an API created by Uber on top of tensorflow that is supposed to make deep learning easy for anyone.
You can operate ludwig from the command line or programmatically using the python API. I tested both.

## Requirements
To use ludwig you need to install the package from pip.

```
pip install ludwig
python -m spacy download en
```

## Explanation
The way ludwig works is that you give it a model_definition.yaml file with all the input features you want to train your model on and your label variable (the variable you want to predict).

The model_definition.yaml file looks like this

```
input_features:
    -
        name: LotArea
        type: numerical
    -
        name: BldgType
        type: category
    -
        name: BsmtFinSF1
        type: numerical
        missing_value_strategy: fill_with_mean
    -
        name: GrLivArea
        type: numerical
    -
        name: GarageType
        type: category

output_features:
    -
        name: SalePrice
        type: numerical
```

To train the model you can simply run
```
ludwig train --data_csv path/to/file.csv --model_definition "{input_features: [{name: doc_text, type: text}], output_features: [{name: class, type: category}]}"
```

To visualize the learning curves you can run
```
ludwig visualize --visualization learning_curves --training_statistics path/to/training_statistics.json
```

To predict on a test file you can run
```
ludwig predict --data_csv path/to/data.csv --model_path /path/to/model
```

You can also use the python api to go through an end to end training example with a prediction.

I ran ludwig on the housing prices kaggle dataset and compared it to some classification algorithms and ludwig generally performed the same without having to do any feature engineering.

To view a full simple end to end training and prediction script you can take a look at the script.
```
ckd.py
```

 I trained a deep learning model with ludwig to predict whether a patient has ckd (chronic kidney disease) or not based on their age and several lab measurements taken. The kidney dataset was downloaded from the uci machine learning dataset repository.
