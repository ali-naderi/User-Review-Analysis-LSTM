import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import json

# Dataset paths
paths = ['dataset/Musical_Instruments_5.json'
         ]

# Initializing RandomUnderSampler
rus = RandomUnderSampler(random_state=1969)

# Initialing arrays
reviews = []
ratings = []

# Looping through the paths
for path in paths:
    print('--Processing first dataset {}th--'.format(path))
    
    # Reading the dataset
    with open(path) as f:
        file_content = f.read()  
    output = [json.loads(line)
              for line in file_content.split("\n") if line.strip() != ""]
    data = pd.DataFrame(output)
    
    # Filtering the dataset
    #data = data[data['verified_purchase'] == 'Y']
    
    # Selecting the two columns
    data = data[['reviewText', 'overall']]
    
    # Dropping rows with nan values
    data = data.dropna()
    
    # Converting the dataframe into numpy arrays
    X = data['reviewText'].values.reshape(-1,1)
    y = pd.to_numeric(data['overall']).values.reshape(-1,1)
    
    # Deleting data to save space
    del data
    
    print('--Sampling dataset--')
    
    # Sampling
    X, y = rus.fit_resample(X,y)
    
    # Appending the data into arrays
    reviews = reviews + X.tolist()
    ratings = ratings + y.tolist()
    
    # Saving space
    del X, y
    
# Initializng a DataFrame
dataset = pd.DataFrame(columns=['reviews', 'ratings'])

# Looping through the ratings to convert into int
newRatings = []
for rate in ratings:
    newRatings.append(int(rate))
   
# Deleting old ratings array
del ratings

# Looping through the reviews to convert into string
newReviews = []
for review in reviews:
    newReviews.append(review[0])
       
# Deleting old reviews array
del reviews
    
# Putting data into the DataFrame
dataset['reviews'] = newReviews
dataset['ratings'] = newRatings

# Deleting reviews and newRatings array
del newReviews, newRatings

# Shuffling the datset
dataset = dataset.sample(frac=1)

# Saving the data as a csv file
dataset.to_csv('dataset/dataset.csv', index=False)
