# 예제 딕셔너리
my_dict = {
    'name': 'Alice',
    'age': 30,
    'city': 'Seoul',
    'job': 'Engineer'
}

# 키 값을 찾는 함수
def find_key_by_expression(dictionary, search_term):
    return [key for key in dictionary if search_term in key]

# 예제 검색 단어: 'a'
search_term = 'a'

# 키 값 찾기
matching_keys = find_key_by_expression(my_dict, search_term)
print(matching_keys)  # 출력: ['name', 'age']
