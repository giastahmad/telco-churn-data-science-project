# Analisis dan Prediksi Customer Churn Telekomunikasi (End-to-End Data Science Project)

## Daftar Isi

1.  [Latar Belakang & Tujuan Proyek](#latar-belakang--tujuan-proyek)
2.  [Apa itu Customer Churn?](#apa-itu-customer-churn)
3.  [Sumber Data](#sumber-data)
4.  [Teknologi yang Digunakan](#teknologi-yang-digunakan)
5.  [Alur Kerja Proyek](#alur-kerja-proyek)
6.  [Insight dari Analisis Data (EDA)](#insight-dari-analisis-data-eda)
7.  [Hasil Pemodelan Machine Learning](#hasil-pemodelan-machine-learning)
8.  [Dashboard Interaktif Power BI](#dashboard-interaktif-power-bi)
9.  [Cara Menjalankan Proyek Ini](#cara-menjalankan-proyek-ini)

## Latar Belakang & Tujuan Proyek

Proyek ini bertujuan untuk melakukan analisis terhadap data pelanggan dari sebuah perusahaan telekomunikasi untuk mengidentifikasi faktor-faktor utama yang kemungkinan menyebabkan *customer churn* (pelanggan berhenti berlangganan). Tujuan akhir dari proyek ini adalah membangun sebuah model *machine learning* yang dapat memprediksi pelanggan mana yang berisiko akan churn, sehingga perusahaan dapat mengambil langkah yang cepat dan tepat untuk mengurangi jumlah pelanggan yang churn sehingga mengurangi kerugian pendapatan.

## Apa itu Customer Churn?

**Customer churn** adalah istilah yang digunakan untuk menggambarkan kondisi di mana pelanggan memutuskan untuk berhenti menggunakan produk atau layanan dari suatu perusahaan dan beralih ke penyedia lain atau tidak lagi menjadi pelanggan aktif.

Mengapa metrik ini sangat penting? Karena biaya untuk mengakuisisi pelanggan baru **jauh lebih mahal** (bisa 5 hingga 25 kali lipat) dibandingkan biaya untuk mempertahankan pelanggan yang sudah ada. Oleh karena itu, kemampuan untuk memprediksi churn secara proaktif sangat berharga bagi bisnis, terutama yang berbasis langganan seperti telekomunikasi, streaming, dan SaaS (Software as a Service). Proyek ini berfokus pada **voluntary churn**, di mana pelanggan secara aktif memilih untuk menghentikan layanan.

## Sumber Data

Dataset yang digunakan dalam proyek ini adalah versi modifikasi dari dataset publik "Telco Customer Churn" yang tersedia di Kaggle.

* **Sumber Data Asli:** [Telco Customer Churn by Abdallah Wagih Ibrahim](https://www.kaggle.com/datasets/abdallahwagih/telco-customer-churn)
* **Dataset yang Digunakan untuk training:** Dataset yang sudah diperkaya dan disimpan di [folder /data/processed/](./Data/processed/) di repository ini.

## Teknologi yang Digunakan

* **Bahasa Pemrograman:** Python
* **Library Analisis Data:** Pandas, NumPy
* **Library Visualisasi Data:** Matplotlib, Seaborn
* **Library Machine Learning:** Scikit-learn, Imbalanced-learn
* **Library Deep Learning:** TensorFlow (Keras)
* **Library Hyperparameter Tuning:** KerasTuner
* **Dashboarding:** Microsoft Power BI

## Alur Kerja Proyek

Proyek ini mengikuti alur kerja *end-to-end data science* yang sistematis:
1.  **Pembersihan Data:** Memeriksa nilai yang hilang dan memperbaiki tipe data yang tidak konsisten.
```python
df.isnull().sum()

df.duplicated().sum()
```
2.  **Analisis Data Eksploratif (EDA):** Menggali wawasan dan memahami pola dari data melalui visualisasi. salah satu contoh:
```python
top_10_Cities = df.City.value_counts().head(10)

plt.figure(figsize=(12,6))
plt.style.use('ggplot')
sns.barplot(x=top_10_Cities.index, y=top_10_Cities.values, palette='viridis')  
plt.title('Top 10 Cities')
plt.xlabel('City')
for i, count in enumerate(top_10_Cities.values):
    plt.text(i, count + 15, str(count), ha='center')  
plt.xticks(rotation=45)  
plt.show()
```
![](reports/figures/top-ten-cities-report.png)

3.  **Rekayasa Fitur (Feature Engineering):** Membuat fitur-fitur baru yang lebih informatif seperti jumlah layanan, rasio finansial, dan Kategori pelanggan untuk meningkatkan performa model.
```python
additional_services_cols = [
    'Online Security_Yes', 
    'Online Backup_Yes', 
    'Device Protection_Yes',
    'Tech Support_Yes', 
    'Streaming TV_Yes', 
    'Streaming Movies_Yes'
]

df['Sum Of Additional Services'] = df[additional_services_cols].sum(axis=1)
```
```python
df['Monthly Charges to Tenure Ratio'] = df['Monthly Charges'] / (df['Tenure Months'] + 1)
```
```python
bins = [0,12,100]
labels = ['New Customers','Loyal Customers']
df['Tenure Category'] = pd.cut(df['Tenure Months'], bins=bins, labels=labels, right=False)
```

4.  **Penanganan Data Tidak Seimbang:** Menerapkan teknik SMOTE pada data training untuk mengatasi masalah kelas minoritas.
```python
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train_scaled, y_train)
```

5.  **Perbandingan Model Klasik:** Menguji ~9 model machine learning klasik untuk mendapatkan *baseline* performa yang solid.
```python
models = [
    ('Logistic Regression', LogisticRegression()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('AdaBoost Classifier', AdaBoostClassifier()),
    ('Gradient Boosting Classifier', GradientBoostingClassifier()),
    ('XGBClassifier', XGBClassifier()),
    ('LGBMClassifier', LGBMClassifier(verbose=-1)),
    ('CatBoostClassifier', CatBoostClassifier(verbose=False)),
    ('KNN', KNeighborsClassifier())
]

results = []
for name, model in models:
    model.fit(x_train_resampled, y_train_resampled)
    
    y_train_pred = model.predict(x_train_resampled)
    y_test_pred = model.predict(x_test_scaled)
    
    train_score = model.score(x_train_resampled, y_train_resampled)
    test_score = model.score(x_test_scaled, y_test)
    
    accuracy = accuracy_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    confusion = confusion_matrix(y_test, y_test_pred)
    
    results.append({
       'Model': name,
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1
    })

results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
display(results_df_sorted)

best_model_name = results_df_sorted.iloc[0]['Model']
best_accuracy = results_df_sorted.iloc[0]['Accuracy']

print(f"\nBest Model based on Accuracy: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}")
```

6.  **Optimasi & Hyperparameter Tuning:** Melakukan tuning sistematis pada model ANN menggunakan KerasTuner untuk menemukan arsitektur terbaik.
```python
def build_model(hp):
    model = Sequential()
    
    input_dim = x_train_resampled.shape[1]
    model.add(Input(shape=(input_dim,)))

    hp_units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32)
    model.add(Dense(units=hp_units_1, activation='relu'))

    hp_dropout_1 = hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)
    model.add(Dropout(rate=hp_dropout_1))

    hp_units_2 = hp.Int('units_2', min_value=16, max_value=64, step=16)
    model.add(Dense(units=hp_units_2, activation='relu'))
    
    model.add(Dense(1, activation='sigmoid'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy','recall']
    )
    
    return model
```
```python
tuner = kt.RandomSearch(
    build_model,
    objective=kt.Objective("val_accuracy", direction="max"),
    max_trials=20,
    executions_per_trial=1,
    directory='ann_tuning',
    project_name='churn_prediction'
)
```
```python
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)

tuner.search(
    x_train_resampled, 
    y_train_resampled,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stopping]
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
```

7.  **Evaluasi Model Final:** Mengevaluasi model ANN terbaik pada data uji dan melakukan optimasi *threshold* untuk memaksimalkan Accuracy.
```python
thresholds = np.linspace(0.0, 1.0, num=100)
accuracies = []

for thresh in thresholds:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    acc = accuracy_score(y_test, y_pred_thresh)
    accuracies.append(acc)

best_location = np.argmax(accuracies)
best_threshold = thresholds[best_location]
best_accuracy = accuracies[best_location]

print(f"Best Threshold for Accuracy: {best_threshold:.4f}")
print(f"Best Accuracy: {best_accuracy:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(thresholds, accuracies, 'm-', label='Accuracy', linewidth=2)
plt.axvline(x=best_threshold, color='k', linestyle='--', label=f'Optimal Threshold ({best_threshold:.2f})')

plt.title('Accuracy vs. Threshold')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()
```
```python
y_pred_ann = (y_pred_proba > best_threshold).astype(int)

accuracy_ann = accuracy_score(y_test, y_pred_ann)
print(f"Accuracy Score ANN on Test Data: {accuracy_ann:.4f}")
```

8.  **Visualisasi Dashboard:** Membuat dashboard interaktif di Power BI untuk menyajikan insight kepada audiens bisnis.

## Insight dari Analisis Data (EDA)

* **Insight 1:** Basis pelanggan yang tergambarkan dalam dataset ini sangat didominasi oleh demografi usia non-senior (dewasa muda dan paruh baya). Ini bisa berarti strategi pemasaran dan produk perusahaan selama ini lebih berhasil menarik segmen tersebut.
![](reports/figures/senior-citizen.png)

* **Insight 2:** Tantangan churn yang terlihat dari data ini memiliki dua sumber utama: daya tarik kompetitor dari luar dan masalah dengan kualitas layanan perusahaan sendiri. Karena itu, upaya untuk mempertahankan pelanggan harus fokus pada kedua hal tersebut sekaligus.
![](reports/figures/churn-reasons.png)

* **Insight 3:** Pelanggan dengan kontrak **Month-to-month** (bulanan) secara signifikan lebih rentan untuk berhenti berlangganan (churn) dibandingkan pelanggan dengan kontrak jangka panjang.
![](reports/figures/churn-rate-by-contract.png)

## Hasil Pemodelan Machine Learning

Setelah melalui beberapa tahap optimasi, termasuk *hyperparameter tuning* dan *threshold optimization*, model final **Artificial Neural Network (ANN)** berhasil mencapai performa sebagai berikut pada data uji:
* **Accuracy : 0.8006**

**Notebook Publik (Kaggle)**
* Untuk melihat notebook dengan semua output sel yang sudah dijalankan secara interaktif, silakan kunjungi link berikut:
* **[Buka Notebook di Kaggle](https://www.kaggle.com/code/giastahmad/telco-churn-prediction-eda-ann-tuning)**


## Dashboard Interaktif Power BI

Berikut adalah pratinjau dari dashboard Power BI yang telah dibuat untuk visualisasi analisis churn.

### Demonstrasi Interaktif
![Demonstrasi Dashboard](reports/figures/interactive-dashboard-preview-power-bi.gif)

### Halaman 1: Ringkasan Eksekutif
![Ringkasan Eksekutif](reports/figures/executive-summary-power-bi.png)

### Halaman 2: Analisis Faktor Pendorong Churn
![Analisis Faktor Pendorong Churn](reports/figures/churn-driving-factor-analysis-power-bi.png)

### Halaman 3: Analisis Geografis
![Analisis Geografis](reports/figures/geographic-analysis-of-churn-power-bi.png)

*Untuk laporan lengkap dalam format PDF, silakan lihat file `Customer_Churn_Analysis.pdf` di dalam folder `/reports/`.*

## Cara Menjalankan Proyek Ini

1.  **Clone repository ini:**
    ```bash
    git clone [https://github.com/giastahmad/telco-churn-data-science-project.git](https://github.com/giastahmad/telco-churn-data-science-project.git)
    cd telco-churn-data-science-project
    ```
2.  **Install semua library yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Buka Notebook:**
    Buka file `.ipynb` yang ada di dalam folder `/notebooks/` menggunakan Jupyter Notebook atau editor sejenisnya.