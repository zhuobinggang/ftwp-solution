import fasttext
from functools import lru_cache
import numpy as np
from common_new import logging
import os

logger = logging.getLogger('fasttext_classifier')

PATH = '/home/taku/Downloads/cog2019_ftwp/trained_models/fasttext_cc.en.300.bin'

OPENABLE = ['screen door', 'fridge', 'barn door', 'plain door', 'wooden door', 
            'sliding patio door', 'patio door', 'frosted-glass door', 'front door', 
            'sliding door', 'commercial glass door', 'fiberglass door', 'toolbox']
AUGMENTED_OPENABLE = ['gate', 'wooden gate', 'drawer', 'cabinet', 'wooden cabinet', 'wardrobe']

UNOPENABLE = ['stove', 'yellow potato', 'workbench', 'north', 'red potato', 
              'showcase', 'orange bell pepper', 'meal', 'sofa', 'block of cheese', 
              'east', 'BBQ', 'bed', 'banana', 'chicken wing', 'parsley', 'white onion', 
              'patio chair', 'red hot pepper', 'red apple', 'yellow bell pepper', 'oven', 
              'red onion', 'toilet', 'pork chop', 'salt', 'black pepper', 'olive oil', 
              'west', 'cookbook', 'chicken leg', 'counter', 'flour', 'patio table', 
              'carrot', 'shelf', 'table', 'purple potato', 'south', 'water', 'cilantro', 'knife']

VALID_OPENABLE = ['door', 'cabinet', 'drawer', 'gate', 'wardrobe', 'window', 'toolbox', 'fridge', 'fiberglass door', 'commercial glass door', 'frosted-glass door', 'screen door', 'front door', 'plain door', 'patio door', 'barn door', 'sliding door', 'wooden door', 'sliding patio door']
VALID_UNOPENABLE = ['cilantro', 'sofa', 'BBQ', 'patio chair', 'black pepper', 'shelf', 'red tuna', 'east', 'chicken wing', 'cookbook', 'banana', 'chicken breast', 'knife', 'west', 'yellow bell pepper', 'pork chop', 'red onion', 'workbench', 'purple potato', 'south', 'water', 'orange bell pepper', 'olive oil', 'lettuce', 'flour', 'patio table', 'counter', 'block of cheese', 'meal', 'green bell pepper', 'bed', 'oven', 'vegetable oil', 'carrot', 'white onion', 'red potato', 'peanut oil', 'red hot pepper', 'yellow potato', 'stove', 'table', 'toilet', 'red bell pepper', 'red apple', 'green apple', 'salt', 'showcase', 'chicken leg', 'parsley', 'north']

OPENABLE_CENTROID_PATH = 'weights/openable_centroid.npy'
UNOPENABLE_CENTROID_PATH = 'weights/unopenable_centroid.npy'

@lru_cache(maxsize=4)
def get_model():
    model = fasttext.load_model(PATH)
    return model

@lru_cache(maxsize=4)
def get_openable_unopenable_centroids():
    print('calculate centroids')
    model = get_model()
    # concated_openable = [entity.replace(' door', '') for entity in OPENABLE]
    concated_openable = OPENABLE + AUGMENTED_OPENABLE
    concated_openable = ['-'.join(entity.split()) for entity in concated_openable]
    print(concated_openable)
    concated_unopenable = ['-'.join(entity.split()) for entity in UNOPENABLE]
    openable_centroid = model.get_sentence_vector(' '.join(concated_openable))
    unopenable_centroid = model.get_sentence_vector(' '.join(concated_unopenable))
    return openable_centroid, unopenable_centroid

def save_centroids():
    a, b = get_openable_unopenable_centroids()
    a.astype(float).tofile(OPENABLE_CENTROID_PATH)
    b.astype(float).tofile(UNOPENABLE_CENTROID_PATH)

@lru_cache(maxsize=4)
def read_centroids():
    print('read centroids from file')
    a = np.fromfile(OPENABLE_CENTROID_PATH, dtype=float)
    b = np.fromfile(UNOPENABLE_CENTROID_PATH, dtype=float)
    return a, b


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@lru_cache(maxsize=128)
def get_word_vector(word):
    model = get_model()
    word_vector = model.get_word_vector(word)
    return word_vector

@lru_cache(maxsize=4)
def get_centroids_from_file_if_possible():
    openable_centroid, unopenable_centroid = None, None
    # 判断weights文件夹中是否存在openable_centroid.npy和unopenable_centroid.npy，如果是则读取，否则计算
    if os.path.exists(OPENABLE_CENTROID_PATH) and os.path.exists(UNOPENABLE_CENTROID_PATH):
        pass
    else:
        save_centroids()
    openable_centroid, unopenable_centroid = read_centroids()
    return openable_centroid, unopenable_centroid

@lru_cache(maxsize=128)
def log_if_never(word, openable_similarity, unopenable_similarity):
    openable = openable_similarity > unopenable_similarity
    prefix = 'openable' if openable else 'unopenable'
    logger.debug(f'{prefix} word: {word}, openable_similarity: {openable_similarity}, unopenable_similarity: {unopenable_similarity}')

def is_openable_entity(word):
    word = '-'.join(word.split())
    word_vector = get_word_vector(word)
    openable_centroid, unopenable_centroid = get_centroids_from_file_if_possible()
    openable_similarity = cosine_similarity(word_vector, openable_centroid)
    unopenable_similarity = cosine_similarity(word_vector, unopenable_centroid)
    # log_if_never(word, openable_similarity, unopenable_similarity)
    if openable_similarity > unopenable_similarity:
        return True
    else:
        return False
    

def check_classifier():
    # openable = OPENABLE + AUGMENTED_OPENABLE
    # unopenable = UNOPENABLE
    openable = VALID_OPENABLE
    unopenable = VALID_UNOPENABLE
    for word in openable:
        if not is_openable_entity(word):
            print(f'{word} should be openable')
            logger.debug(f'{word} should be openable')
    for word in unopenable:
        if is_openable_entity(word):
            print(f'{word} should be unopenable')
            logger.debug(f'{word} should be unopenable')