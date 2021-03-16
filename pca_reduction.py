import pickle


def pca_reduction(filename, features):
    loaded_model_pca = pickle.load(open('models/'+filename, 'rb'))
    return loaded_model_pca.transform(features)