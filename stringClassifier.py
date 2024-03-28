def classify_strings(string_array):
    classifications = {}
    result = []

    # Her farklı string için bir sınıf numarası oluştur
    class_num = 1
    for string in string_array:
        if string not in classifications:
            classifications[string] = class_num
            class_num += 1

    # Her string için sınıf numarasını al
    for string in string_array:
        result.append(classifications[string])

    return result

