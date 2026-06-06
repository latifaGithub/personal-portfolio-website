---
layout: post
title: Building a Scalable Visual Search Engine Using Self-Supervised Vision Transformers
image: "/posts/dl-search-engine-title-img.png"
description: "I built a deep learning–powered visual product search system that allows users to upload an image and retrieve visually similar products from a catalog using VGG16 embeddings and cosine similarity."
---

In this project, I built a deep learning–powered visual product search system that allows users to upload an image and retrieve visually similar products from a catalog.

Traditional e-commerce search relies heavily on text queries, but customers often struggle to describe products using keywords. Visual search solves this problem by enabling users to search with images instead of text, dramatically improving product discovery.

This project demonstrates how deep visual embeddings combined with vector similarity search can power scalable visual search systems similar to those used by companies such as Amazon, Pinterest, and Google Lens.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Sample Data Overview](#sample-data-overview)
3. [Deep Learning Embeddings for Visual Search](#deep-learning-embeddings-for-visual-search)
4. [Feature Extraction Architecture](#feature-extraction-architecture)
5. [Image Preprocessing and Feature Encoding](#image-preprocessing-and-feature-encoding)
6. [Vector Similarity Search](#vector-similarity-search)
7. [Results and Evaluation](#results-and-evaluation)
8. [Scaling Considerations and Future Improvements](#scaling-considerations-and-future-improvements)

---

## Project Overview

### Context

Many e-commerce platforms struggle with product discoverability. Customers frequently know what a product looks like but cannot easily describe it in words.

During a customer feedback analysis exercise, I identified a common complaint: customers were unable to find visually similar products within the catalog even though those items existed. As a result, they often purchased more expensive alternatives from other platforms.

To address this problem, I designed a visual search system that allows customers to upload an image of a product and retrieve visually similar items from the store catalog.

The goal of the system is to:

- Convert product images into deep learning feature embeddings
- Store those embeddings in a vector search index
- Compare query images against catalog embeddings
- Retrieve the most visually similar products in real time

---

## Sample Data Overview

For this proof-of-concept implementation, I used a dataset containing 300 product images from the women's footwear category.

Each image represents a product available in the catalog.

The objective is to extract meaningful visual features from these images and store them as embedding vectors. When a new image is submitted as a query, the system compares its embedding against the stored embeddings to identify the most similar products.

![Image examples](/img/posts/search-engine-image-examples.png "Deep Learning Search Engine - Image Examples")

---

## Deep Learning Embeddings for Visual Search

Modern visual search systems rely on feature embeddings produced by deep neural networks.

Instead of predicting a specific class label, these models learn a dense vector representation of visual characteristics, including:

- shapes
- textures
- colors
- structural patterns

Images with similar visual properties generate embedding vectors that are close together in high-dimensional space. This allows similarity search to be performed using vector distance metrics.

---

## Feature Extraction Architecture

To generate visual embeddings, I used a pretrained convolutional neural network (VGG16) trained on the ImageNet dataset.

Rather than using the network for classification, I repurposed it as a feature extraction model using transfer learning.

I modified the architecture by replacing the final pooling layer with a Global Average Pooling layer, which converts the final convolutional feature maps into a single 512-dimensional feature vector.

This vector represents the semantic visual characteristics of the image and serves as the embedding used for similarity search.

Using transfer learning allowed me to leverage rich visual features learned from over one million images, eliminating the need to train a deep model from scratch.

![VGG16 Architecture](/img/posts/vgg16-architecture.png "VGG16 Architecture")

---

## Image Preprocessing and Feature Encoding

Before passing images through the neural network, I implemented a preprocessing pipeline that:

1. Resizes images to 224 × 224 pixels
2. Converts images into numerical arrays
3. Adds the batch dimension required by the model
4. Applies ImageNet normalization preprocessing
5. Passes the image through the feature extraction model

The output of the model is a 512-dimensional embedding vector that represents the visual characteristics of the image.

I generated embeddings for all product images in the dataset and stored them for later use during search queries.

```python
# image pre-processing function
def preprocess_image(filepath):
    image = load_img(filepath, target_size=(img_width, img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

# image featurisation function
def featurise_image(image):
    feature_vector = model.predict(image)
    return feature_vector
```

---

## Vector Similarity Search

Once embeddings were generated, I implemented a vector similarity search pipeline.

When a user uploads a query image:

1. The system preprocesses the image.
2. The image is converted into a feature embedding using the same deep learning model.
3. The embedding is compared against all catalog embeddings.

To measure similarity, I used cosine similarity, which calculates the angular distance between vectors in high-dimensional space. Images with smaller cosine distances are considered more visually similar.

To efficiently retrieve the closest matches, I used a nearest neighbor search algorithm to return the top-K most similar products.

---

## Results and Evaluation

I tested the system using multiple query images from the footwear dataset.

**Search Image**

![Search 1](/img/posts/search-engine-search1.jpg "Search Image 1")

**Search Results**

![Search Results 1](/img/posts/search-engine-search1-results.png "Search Results 1")

**Search Image**

![Search 2](/img/posts/search-engine-search2.jpg "Search Image 2")

**Search Results**

![Search Results 2](/img/posts/search-engine-search2-results.png "Search Results 2")

In each case, the system successfully retrieved products that were visually similar to the query image in terms of shape, color patterns, design features, and product structure.

Even when background conditions or lighting varied between images, the model was able to capture the underlying visual characteristics of the products.

These results demonstrate that deep learning embeddings can effectively power visual search functionality in product catalogs.

---

## Scaling Considerations and Future Improvements

This implementation was designed as a proof-of-concept prototype, but several improvements would be required for production deployment.

### Vector Databases

For large-scale systems with millions of images, embeddings should be stored in vector databases such as FAISS, Pinecone, Milvus, or Elasticsearch vector search. These systems enable sub-second similarity search across massive embedding indexes.

### Modern Vision Models

Although VGG16 works well for this prototype, modern systems typically use Vision Transformer–based architectures such as CLIP, ViT, or DINOv2. These models produce higher-quality semantic embeddings and significantly improve retrieval accuracy.

### Real-Time Catalog Updates

A production system would require pipelines that generate embeddings for new product images, remove embeddings for out-of-stock products, and continuously update the vector index.

### Microservice Architecture

A scalable architecture for visual search could follow this pipeline:

```
Image Upload
↓
Feature Encoder (Deep Learning Model)
↓
Vector Database
↓
Similarity Search
↓
Product Ranking API
```

This architecture allows visual search to integrate seamlessly into modern e-commerce recommendation systems.
