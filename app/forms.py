from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField, StringField, IntegerField
from wtforms.validators import DataRequired, NumberRange

class ClassifierForm(FlaskForm):
    classifier = SelectField('Escolha o Classificador:', choices=[
        ('KNN', 'KNN'),
        ('SVM', 'SVM'),
        ('DT', 'Decision Tree'),
        ('RF', 'Random Forest')
    ])
    submit = SubmitField('Enviar')

class KNNForm(FlaskForm):
    n_neighbors = IntegerField('Vizinhos', validators=[DataRequired(), NumberRange(min=1)])
    submit = SubmitField('Treinar KNN')

class SVMForm(FlaskForm):
    kernel = SelectField('Kernel', choices=[('linear', 'Linear'), ('poly', 'Polinomial')])
    degree = IntegerField('Grau', validators=[NumberRange(min=1)], default=3)
    submit = SubmitField('Treinar SVM')

class DTForm(FlaskForm):
    max_depth = IntegerField('Profundidade Máxima', validators=[NumberRange(min=1)])
    submit = SubmitField('Treinar DT')

class RFForm(FlaskForm):
    n_estimators = IntegerField('Número de Árvores', validators=[DataRequired(), NumberRange(min=1)])
    max_depth = IntegerField('Profundidade Máxima', validators=[NumberRange(min=1)], default=None)
    submit = SubmitField('Treinar RF')