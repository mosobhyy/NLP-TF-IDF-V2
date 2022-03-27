import numpy as np
import pandas as pd

from scipy.spatial import distance

from utility import text_preprocessing, tokenization, create_domain_count, create_domain_tf, create_idf, \
    create_domain_tf_idf

with open('football-train') as f:
    football_train = f.read()

with open('os-train') as f:
    os_train = f.read()

with open('football-test') as f:
    football_test = f.read()

with open('os-test') as f:
    os_test = f.read()

football_train = text_preprocessing(football_train)
os_train = text_preprocessing(os_train)
domains_dict = football_train + ' ' + os_train
domains_dict = tokenization(domains_dict)

train_dict = test_dict = domains_dict

# Training data
train_dict = create_domain_count(train_dict, football_train, os_train)

train_dict = create_domain_tf(train_dict)

train_dict = create_idf(train_dict)

train_dict = create_domain_tf_idf(train_dict)

df_train = pd.DataFrame.from_dict(train_dict, orient='index')

df_train = df_train.rename(columns={0: 'COUNT1', 1: 'COUNT2', 2: 'TF1', 3: 'TF2', 4: 'IDF', 5: 'TF-IDF1', 6: 'TF-IDF2'})

print(df_train.head(10))

# Test data

test_dict = create_domain_count(test_dict, football_test)

test_dict = create_domain_tf(test_dict)

# Calculate total domains
domains_values = list(train_dict.values())
total_domains = int((len(domains_values[0]) - 1) / 3)

# Copy idf of train dictionary
for word, values in test_dict.items():
    test_dict[word].append(train_dict[word][-1 - total_domains])

test_dict = create_domain_tf_idf(test_dict)

df_test = pd.DataFrame.from_dict(test_dict, orient='index')

df_test = df_test.rename(columns={0: 'COUNT', 1: 'TF', 2: 'IDF', 3: 'TF-IDF'})

print(df_test.head(10))

# Calculate sum of every tf-idf column in train dictionary
train_tf_idf_sum = np.zeros(total_domains)

for word, values in train_dict.items():
    for i in range(total_domains):
        train_tf_idf_sum[i] += train_dict[word][total_domains * -1 + i]

# Calculate sum tf-idf column in test dictionary
test_tf_idf_sum = 0
for word, values in test_dict.items():
    test_tf_idf_sum += test_dict[word][-1]

# Calculate euclidean distance between train TF-IDFs and test IF-IDF
football_distance = distance.euclidean(train_tf_idf_sum[0], test_tf_idf_sum)
os_distance = distance.euclidean(train_tf_idf_sum[1], test_tf_idf_sum)

# The nearest the winner
print(football_distance, os_distance)
if football_distance < os_distance:
    print("Football Domain")
else:
    print("OS Domain")
