import io
import joblib
from sklearn import svm, datasets


def save(bytes_container: io.BytesIO):
    """

    :param bytes_container:
    :return:
    """
    clf = svm.SVC()
    X, y = datasets.load_iris(return_X_y=True)
    clf.fit(X, y)

    #dump model
    svc_byte_io = io.BytesIO()
    joblib.dump(clf, svc_byte_io)

    pack = {'class_dict':  {'a': 12231},
            'model': svc_byte_io.getvalue()}

    joblib.dump(pack, bytes_container)


def load(bytes_container: io.BytesIO) -> (svm.SVC, dict):
    """

    :param bytes_container:
    :return:
    """
    pack = joblib.load(bytes_container)
    class_dict = pack['class_dict']
    clf = joblib.load(io.BytesIO(pack['model']))
    return clf, class_dict


if __name__ == "__main__":

    #in memory example
    bytes_container = io.BytesIO()
    save(bytes_container)
    clf, class_dict = load(io.BytesIO(bytes_container.getvalue()))
    print(clf.get_params())
    print(class_dict)

    #save into file example
    with open("mymodel", "wb") as f:        
        save(f)

    #load from file
    with open("mymodel", "rb") as f:
        clf, class_dict = load(f)

        print(clf.get_params())
        print(class_dict)
