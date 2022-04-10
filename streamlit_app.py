import math

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from faiss import index_factory, METRIC_INNER_PRODUCT, normalize_L2
from sentence_transformers import models, SentenceTransformer

st.set_page_config(layout="wide")

class Classifier:
    def __init__(self, labels, embeddings):
        self.labels = labels
        self.embeddings = embeddings
        normalize_L2(self.embeddings)

        COUNT = embeddings.shape[0]
        DIMENSIONS = embeddings.shape[1]

        storage = "Flat"
        cells = min(round(4 * math.sqrt(COUNT)), int(COUNT / 39))
        cells = max(cells, 5)

        params = f"IVF{cells},{storage}"
        print(f'params: {params}')
        self.index = index_factory(DIMENSIONS, params, METRIC_INNER_PRODUCT)
        self.index.train(embeddings)
        self.index.add(embeddings)
        self.index.nprobe = 5
        print(f'nprobe: {self.index.nprobe}')

    def predict(self, query_vectors, count, prob_threshold):
        normalize_L2(query_vectors)
        all_probs, all_ids = self.index.search(query_vectors, count)

        data = []

        for probs, ids in zip(all_probs, all_ids):
            data.append([(self.labels[id], prob)
                         for id, prob in zip(ids, probs) if id >= 0 and prob > prob_threshold])

        return data

    def predict_single(self, query_vector, count):
        query_vectors = np.asarray([query_vector])
        return self.predict(query_vectors, count)[0]


st.title('OpenAI Clip zero-shot image classification')

with st.sidebar:
    st_labels = st.text_area('Labels:',
                             'naked\nbikini\nswimwear\nanime\nrape\nboobs\nvagina\nblood\ndead\ngun\nfighting\nbedroom\nbeach\nmeme\nwar\nsports',
                             height=500)


@st.cache
def load_text_model():
    return SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')


@st.cache(allow_output_mutation=True)
def load_image_model():
    clip_model = models.CLIPModel()
    return SentenceTransformer(modules=[clip_model])


# label
@st.cache
def load_classifier(labels):
    model = load_text_model()
    labels = list(filter(None, labels))
    labels2 = [f'{l} {l} {l}' for l in labels]
    labels_embeddings = model.encode(labels2)

    return Classifier(labels, labels_embeddings)


field_labels = st_labels.split('\n')
ts = load_classifier(field_labels)

# test
st_image_urls = st.text_area('Image URLs:',
                             'https://pbs.twimg.com/media/E7Ni6RFXIAIwju_.jpg\nhttps://pbs.twimg.com/media/E7Niw-MXIA88N8c.jpg\nhttps://pbs.twimg.com/media/BBuCZgpCUAAZqnX.jpg',
                             height=300)
field_image_urls = st_image_urls.split('\n')


@st.cache
def load(url):
    print(url)
    try:
        image = Image.open(requests.get(url, stream=True).raw)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.thumbnail((320, 240))
        return image
    except:
        print(f"Error loading image: {url}")
        return None


def create_embeddings(field_image_urls):
    urls = []
    images = []

    for url in field_image_urls:
        im = load(url)
        if im is not None:
            urls.append(url)
            images.append(im)

    image_model = load_image_model()
    return urls, images, image_model.encode(images)


if st.button('Classify'):
    st.header('Predictions')

    data_load_state = st.text('Loading images...')
    image_urls, images, image_embeddings = create_embeddings(field_image_urls)

    # predict
    data_load_state.text('Classifying...')
    predictions = ts.predict(image_embeddings, count=2, prob_threshold=0.225)

    c = 0
    labels = []
    for p in predictions:
        keys = [k for k, v in p]
        keys.sort()
        keys = ' & '.join(keys)
        c = c + 1
        labels.append(keys)

    data = []
    for i, l in zip(image_urls, labels):
        data.append({
            'url': i,
            'labels': l
        })

    df = pd.DataFrame(data)
    st.dataframe(df)

    data_load_state.text('')

    # show images
    images_np = np.array(images)
    for l in set(labels):
        df2 = df[df.labels == l]
        i = images_np[df2.index].tolist()

        l = l if len(l) > 0 else 'No categories found'
        st.text(f' ')
        st.text(f'Photos with: {l}:')
        st.image(i, width=200)