from flask import flash, redirect, render_template, request, url_for
from sklearn.model_selection import train_test_split

from app import app
from .forms import ClassifierForm, KNNForm, SVMForm, DTForm, RFForm
from machine import carregar, treinar

@app.route('/classifier_', methods=['GET', 'POST'])
def classifier_():
    classifier = request.args.get('classifier', 'KNN')
    form = None
    parametros = {}
    
    if classifier == 'KNN':
        form = KNNForm(request.form)
        if form.validate_on_submit():
            parametros = {'n_neighbors': form.n_neighbors.data}
    elif classifier == 'SVM':
        form = SVMForm(request.form)
        if form.validate_on_submit():
            parametros = {'kernel': form.kernel.data, 'degree': form.degree.data}
    elif classifier == 'DT':
        form = DTForm(request.form)
        if form.validate_on_submit():
            parametros = {'max_depth': form.max_depth.data}
    elif classifier == 'RF':
        form = RFForm(request.form)
        if form.validate_on_submit():
            parametros = {'n_estimators': form.n_estimators.data, 'max_depth': form.max_depth.data}

    if request.method == 'POST' and form.validate():
        X, y = carregar()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        image_path = treinar(classifier, parametros, X_train, y_train, X_test, y_test)
        print(image_path)

        flash('Modelo treinado com sucesso!')
        return render_template('predict.html', form=form, classifier=classifier, matrix_image=image_path)

    return render_template('predict.html', form=form, classifier=classifier)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = ClassifierForm()
    if form.validate_on_submit():
        classifier = form.classifier.data
        return redirect(url_for('classifier_', classifier=classifier))

    return render_template('index.html', form=form)

