import streamlit as st
import pandas as pd
import math


st.set_page_config(page_title="Klasifikasi Nasabah", page_icon=":money_with_wings:")
# Fungsi-fungsi utama
def calculate_prior_probabilities(data):
    total_samples = len(data)
    class_counts = {}
    for sample in data:
        label = sample[-1]
        class_counts[label] = class_counts.get(label, 0) + 1
    return {label: count / total_samples for label, count in class_counts.items()}

def calculate_likelihoods_categorical(data, feature_indices, class_labels):
    likelihoods = {}
    for label in class_labels:
        filtered_data = [sample for sample in data if sample[-1] == label]
        total_count = len(filtered_data)
        feature_likelihoods = {}
        for index in feature_indices:
            feature_counts = {}
            for sample in filtered_data:
                value = sample[index]
                feature_counts[value] = feature_counts.get(value, 0) + 1
            feature_likelihoods[index] = {value: count / total_count for value, count in feature_counts.items()}
        likelihoods[label] = feature_likelihoods
    return likelihoods

def calculate_gaussian_probability(x, mean, std):
    if std == 0:
        return 1e-6 if x != mean else 1.0
    exponent = math.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

def calculate_likelihoods_numerical(data, feature_indices, class_labels):
    likelihoods = {}
    for label in class_labels:
        filtered_data = [sample for sample in data if sample[-1] == label]
        feature_stats = {}
        for index in feature_indices:
            values = [float(sample[index]) for sample in filtered_data]
            mean = sum(values) / len(values)
            std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
            feature_stats[index] = (mean, std)
        likelihoods[label] = feature_stats
    return likelihoods

def naive_bayes_predict(sample, prior_probs, likelihoods_categorical, likelihoods_numerical, numerical_indices, class_labels):
    probabilities = {}
    for label in class_labels:
        probability = prior_probs[label]
        for index, value in enumerate(sample):
            if index in numerical_indices:
                mean, std = likelihoods_numerical[label][index]
                probability *= calculate_gaussian_probability(float(value), mean, std)
            else:
                probability *= likelihoods_categorical[label][index].get(value, 1e-6)
        probabilities[label] = probability
    return max(probabilities, key=probabilities.get)

def calculate_attribute_probabilities(data, numerical_indices, categorical_indices, class_labels):
    attribute_probabilities = {}

    # Probabilitas untuk fitur kategorikal
    for index in categorical_indices:
        attribute_probabilities[index] = {}
        for label in class_labels:
            filtered_data = [sample for sample in data if sample[-1] == label]
            value_counts = {}
            for sample in filtered_data:
                value = sample[index]
                value_counts[value] = value_counts.get(value, 0) + 1

            total_samples = len(filtered_data)
            attribute_probabilities[index][label] = {
                value: count / total_samples
                for value, count in value_counts.items()
            }

    # Rentang kustom untuk fitur numerik
    custom_ranges = {
        # Format: index: [(min, max, label), ...]
        0: [  # Usia
            (17, 22, "17-22"),
            (23, 28, "23-28"),
            (29, 34, "29-34"),
            (35, 40, "35-40"),
            (41, float('inf'), ">40")
        ],
        7: [  # Tenor
            (3, 4, "3-4 bulan"),
            (5, 6, "5-6 bulan"),
            (7, 8, "7-8 bulan"),
            (9, 10, "9-10 bulan"),
            (11, 12, "11-12 bulan")
        ]
    }

    # Probabilitas untuk fitur numerik dengan rentang kustom
    for index in numerical_indices:
        attribute_probabilities[index] = {}
        for label in class_labels:
            filtered_data = [float(sample[index]) for sample in data if sample[-1] == label]

            # Tentukan rentang berdasarkan kustom atau bagi default 4 kategori
            if index in custom_ranges:
                ranges = custom_ranges[index]
            else:
                min_val, max_val = min(filtered_data), max(filtered_data)
                bin_size = (max_val - min_val) / 4
                ranges = [
                    (min_val, min_val + bin_size, "Sangat Rendah"),
                    (min_val + bin_size, min_val + 2*bin_size, "Rendah"),
                    (min_val + 2*bin_size, min_val + 3*bin_size, "Sedang"),
                    (min_val + 3*bin_size, float('inf'), "Tinggi")
                ]

            # Hitung jumlah dalam setiap rentang
            range_counts = {label: 0 for _, _, label in ranges}
            for value in filtered_data:
                for min_val, max_val, range_label in ranges:
                    if min_val <= value <= max_val:
                        range_counts[range_label] += 1
                        break

            # Konversi ke probabilitas
            total_samples = len(filtered_data)
            attribute_probabilities[index][label] = {
                range_label: count / total_samples
                for range_label, count in range_counts.items()
            }

    return attribute_probabilities


# Fungsi tambahan untuk confusion matrix
def calculate_confusion_matrix(data, prior_probs, likelihoods_categorical, likelihoods_numerical, numerical_indices, class_labels):
    confusion_matrix = {label: {l: 0 for l in class_labels} for label in class_labels}
    for sample in data:
        true_label = sample[-1]
        predicted_label = naive_bayes_predict(sample[:-1], prior_probs, likelihoods_categorical, likelihoods_numerical, numerical_indices, class_labels)
        confusion_matrix[true_label][predicted_label] += 1
    return confusion_matrix

def calculate_accuracy(confusion_matrix):
    correct_predictions = sum(confusion_matrix[label][label] for label in confusion_matrix)
    total_predictions = sum(sum(row.values()) for row in confusion_matrix.values())
    return (correct_predictions / total_predictions) * 100

# Streamlit UI
st.title("Klasifikasi Kelayakan Nasabah💰")

# File upload
uploaded_file = st.file_uploader("Upload data training Anda📤", type=["xlsx"])

if uploaded_file:
    st.subheader("Data Training Preview👀")
    dataset = pd.read_excel(uploaded_file)
    st.write(dataset.head())

    # Konversi dataset ke list
    data = dataset.values.tolist()
    numerical_indices = [0, 4, 6, 7]  # Indeks data numerik
    categorical_indices = [i for i in range(len(data[0]) - 1) if i not in numerical_indices]
    class_labels = list(set(sample[-1] for sample in data))

    # Hitung prior probabilities, likelihoods, dan probabilitas atribut
    prior_probs = calculate_prior_probabilities(data)
    likelihoods_categorical = calculate_likelihoods_categorical(data, categorical_indices, class_labels)
    likelihoods_numerical = calculate_likelihoods_numerical(data, numerical_indices, class_labels)
    attribute_probabilities = calculate_attribute_probabilities(data, numerical_indices, categorical_indices, class_labels)

    # Tampilkan probabilitas atribut
    st.subheader("Probabilitas Atribut untuk Setiap Kelas")
    
    for index, atribut in enumerate(dataset.columns[:-1]):
        st.write(f"### {atribut}")
        
        if index in numerical_indices:
            # Untuk fitur numerik, tampilkan probabilitas setiap rentang
            for label in class_labels:
                st.write(f"Kelas {label}:")
                probabilities = attribute_probabilities[index][label]
                for value, prob in probabilities.items():
                    st.write(f"- {value}: {prob:.2%}")
        else:
            # Untuk fitur kategorikal, tampilkan probabilitas setiap value
            for label in class_labels:
                st.write(f"Kelas {label}:")
                probabilities = attribute_probabilities[index][label]
                for value, prob in probabilities.items():
                    st.write(f"- {value}: {prob:.2%}")
        
        st.write("---")

    st.subheader("Confusion Matrix dan Akurasi")
    confusion_matrix = calculate_confusion_matrix(data, prior_probs, likelihoods_categorical, likelihoods_numerical, numerical_indices, class_labels)
    st.write("Confusion Matrix:")
    st.write(pd.DataFrame(confusion_matrix))

    accuracy = calculate_accuracy(confusion_matrix)
    st.success(f"Akurasi Model: {accuracy:.2f}%")

    st.subheader("Prediksi dan Evaluasi")
    st.subheader("Masukkan Data Nasabah untuk Prediksi")
    sample_to_predict = []
    input_prompts = [
        "Usia (numerik)", "Jenis Kelamin (L/P)", "Status Perkawinan", "Profesi",
        "Penghasilan (numerik)", "Status Pinjaman", "Nilai Pinjam (numerik)", "Tenor (numerik)"
    ]

    for prompt in input_prompts:
        value = st.text_input(f"{prompt}:")
        sample_to_predict.append(value)

    if st.button("Prediksi Kelas"):
        # Lakukan prediksi
        predicted_class = naive_bayes_predict(sample_to_predict, prior_probs, likelihoods_categorical, likelihoods_numerical, numerical_indices, class_labels)
        st.success(f"Kelas yang Diprediksi: {predicted_class}")

    
