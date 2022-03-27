import math
import re

import numpy as np


def text_preprocessing(text):
    # Setting every word to lower
    text = text.lower()

    # Removing punctuations
    text = re.sub(r'[()\[\]{}!-/–;:\'",<>./?@#$%^&*_“~\\]', ' ', text)

    # Removing digits
    text = re.sub(r'\d', '', text)

    # Removing sequential whitespaces
    text = re.sub(r'\s+', ' ', text)

    # Removing leading and trailing whitespaces
    text = text.strip()

    return text


def tokenization(text) -> list:
    text = text.split()
    text = list(set(text))
    text.sort()

    return text


def create_domain_count(text, *domains) -> dict:
    domain_count = {}
    for word in text:
        domain_count[word] = []
        for i in range(len(domains)):
            # Split used to avoid count every character in a word   (EX: avoid count 'a' in great)
            split_sentence = domains[i].split()
            domain_count[word].append(split_sentence.count(word))

    return domain_count


def create_domain_tf(domains_dict) -> dict:
    domains_values = list(domains_dict.values())
    total_counts = np.zeros(len(domains_values[0]))

    # Calculate total count of every single domain
    for i in range(len(domains_values)):
        for j in range(len(total_counts)):
            total_counts[j] += domains_values[i][j]

    for word in domains_dict.keys():
        for i in range(len(total_counts)):
            domains_dict[word].append(domains_dict[word][i] / total_counts[i])

    return domains_dict


def create_idf(domains_dict):
    domains_values = list(domains_dict.values())
    total_domains = int(len(domains_values[0]) / 2)

    for word, values in domains_dict.items():
        # Calculate how many docs has the word
        total_docs_count = len([i for i in values[:total_domains] if i > 0])

        domains_dict[word].append(math.log(total_domains / total_docs_count, 10))  # Log of base 10

    return domains_dict


def create_domain_tf_idf(domains_dict):
    domains_values = list(domains_dict.values())
    total_domains = int(len(domains_values[0]) / 2)

    for word in domains_dict.keys():
        for i in range(total_domains):
            domains_dict[word].append(
                domains_dict[word][i + total_domains] * domains_dict[word][total_domains * 2])

    return domains_dict
