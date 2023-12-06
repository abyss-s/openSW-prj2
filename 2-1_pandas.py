# Project #2-1 Data analysis with pandas
import pandas as pd

file_path = '2019_kbo_for_kaggle_v2.csv'
df = pd.read_csv(file_path)
print('\n2023-2 OSS Project #2\nBaseball Data Analysis\n')


# 1. 2015~ 2018년동안 안타, 타율, 홈런, 출루율 별 상위 10명 출력
def print_top10_players(df, year_range=(2015, 2018), n=10):
    print('[과제 2-1-1번]')

    for year in range(year_range[0], year_range[1] + 1):
        df_year = df[df['year'] == year]

        # 안타 상위 10명
        top10_hits = df_year.nlargest(n, 'H')
        print(f'\n---{year}년도 안타 상위 10명---')
        print(top10_hits[['batter_name', 'H']])

        # 타율 상위 10명
        top10_avg = df_year.nlargest(n, 'avg')
        print(f'\n---{year}년도 타율 상위 10명---')
        print(top10_avg[['batter_name', 'avg']])

        # 홈런 상위 10명
        top10_hr = df_year.nlargest(n, 'HR')
        print(f'\n---{year}년도 홈런 상위 10명---')
        print(top10_hr[['batter_name', 'HR']])

        # 출루율 상위 10명
        top10_obp = df_year.nlargest(n, 'OBP')
        print(f'\n---{year}년도 출루율 상위 10명---')
        print(top10_obp[['batter_name', 'OBP']])


# 2. 2018년에 승리 기여도가 가장 높은 선수 포지선 별로 출력
def print_highest_war_players_by_position(df, year=2018, position='cp', war='war'):
    df_year = df[df['year'] == year]
    hw_players = df_year.sort_values(by=war, ascending=False).groupby(position).head(1)
    print('\n[과제 2-1-2번]')
    print(hw_players[['batter_name', position, war]])


# 3. 연봉과 가장 높은 상관 관계를 가진 열 출력
def print_highest_correlation_with_salary(df, col='salary'):
    df_by_numeric = df.select_dtypes(include=['float64', 'int64'])
    correlation_results = df_by_numeric.corr()[col].drop(col).abs()
    highest_correlation = correlation_results.idxmax()
    print('\n[과제 2-1-3번]')
    print(f'열 이름: {highest_correlation}')
    print(f'상관 계수: {correlation_results[highest_correlation]}')


# 각 함수 호출
print_top10_players(df)
print_highest_war_players_by_position(df)
print_highest_correlation_with_salary(df)
