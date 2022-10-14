# Data dir

The three datasets can be found and downloaded from the url:

https://grouplens.org/datasets/movielens/1m/

https://jmcauley.ucsd.edu/data/amazon/

https://www.instacart.com/datasets/grocery-shopping-2017

The organizations of the data dir should be as follow:

    . Amazon
    ├── metadata.json                   # the metadata for amazon items
    ├── review_Electronics_5.json       # the pre-processed customer review data
    
    . ml-1m
    ├── movies.dat                   # the metadata for the movies
    ├── ratings.dat                  # the viewers' sequential ratings
    
    . instacart_2017_05_01
    ├── order_products__train.csv    # the preprocessed grocery basket data
    ├── products.csv                 # the meta information of the products
    
