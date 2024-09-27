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


import re

# 예제 문자열
text = "The quick brown fox jumps over the lazy dog."

# 검색할 특정 문자열
search_term = "fox"

# 정규 표현식 패턴
pattern = re.escape(search_term)  # 특정 문자열을 정규 표현식 패턴으로 변환

# 정규 표현식을 사용하여 문자열 검색
matches = re.findall(pattern, text)

# 검색 결과 출력
print(matches)  # 출력: ['fox']
