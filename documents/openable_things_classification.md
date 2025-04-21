## openable and unopenable entities

Train set:

* openable (13): {'screen door', 'fridge', 'barn door', 'plain door', 'wooden door', 'sliding patio door', 'patio door', 'frosted-glass door', 'front door', 'sliding door', 'commercial glass door', 'fiberglass door', 'toolbox'}
* unopenable (42): {'stove', 'yellow potato', 'workbench', 'north', 'red potato', 'showcase', 'orange bell pepper', 'meal', 'sofa', 'block of cheese', 'east', 'BBQ', 'bed', 'banana', 'chicken wing', 'parsley', 'white onion', 'patio chair', 'red hot pepper', 'red apple', 'yellow bell pepper', 'oven', 'red onion', 'toilet', 'pork chop', 'salt', 'black pepper', 'olive oil', 'west', 'cookbook', 'chicken leg', 'counter', 'flour', 'patio table', 'carrot', 'shelf', 'table', 'purple potato', 'south', 'water', 'cilantro', 'knife'}

Valid set:

* {'toolbox', 'fridge', 'fiberglass door', 'commercial glass door', 'frosted-glass door', 'screen door', 'front door', 'plain door', 'patio door', 'barn door', 'sliding door', 'wooden door', 'sliding patio door'}
* {'cilantro', 'sofa', 'BBQ', 'patio chair', 'black pepper', 'shelf', 'red tuna', 'east', 'chicken wing', 'cookbook', 'banana', 'chicken breast', 'knife', 'west', 'yellow bell pepper', 'pork chop', 'red onion', 'workbench', 'purple potato', 'south', 'water', 'orange bell pepper', 'olive oil', 'lettuce', 'flour', 'patio table', 'counter', 'block of cheese', 'meal', 'green bell pepper', 'bed', 'oven', 'vegetable oil', 'carrot', 'white onion', 'red potato', 'peanut oil', 'red hot pepper', 'yellow potato', 'stove', 'table', 'toilet', 'red bell pepper', 'red apple', 'green apple', 'salt', 'showcase', 'chicken leg', 'parsley', 'north'}


## 2025.4.21 `fasttext_classifier.check_classifier()` results

``` txt
patio chair should be unopenable
workbench should be unopenable
patio table should be unopenable
```

We will allow more openable entities but not vice versa.