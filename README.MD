prep:

https://www.kaggle.com/code/gordonyip/calibrating-and-binning-ariel-data
https://www.kaggle.com/code/ilu000/ariel25-quick-data-prep


eda:

https://www.kaggle.com/code/lordpatil/perfect-eda-doesn-t-exist/notebook


data generation

https://www.kaggle.com/competitions/ariel-data-challenge-2025/discussion/587936

2024 ariel

https://www.kaggle.com/code/skril31/adc-2024-space-coders


Starter Solution : 1D + 2D CNN
https://www.kaggle.com/code/ahsuna123/starter-solution-1d-2d-cnn

public solution
https://www.kaggle.com/code/vitalykudelya/neurips-non-ml-transit-curve-fitting?scriptVersionId=250312271

pip wheel -w ./packages -r requirements.txt

    pip install -U kaggle==1.6.17
    kaggle competitions download -c ariel-data-challenge-2025


1. Ненормированные данные по транзиту -
2. Равномерно распределенные таймслоты (+ п.1) -
3. batchnorm перед конволюцией +

5. Увеличить число таймслотов -
6. Повысить ширину airs -
7. Попеременный апдейт весов +

4. Повысить разрешение -
8. Планомерно отключать предобработку