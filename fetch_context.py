import pandas as pd
from metapub import PubMedFetcher
import time
import os

def fetch_and_save_articles():
    # 讀取原始 Excel 檔案
    print("正在讀取 Excel 檔案...")
    df = pd.read_excel('CDC_datasetML_modify.xlsx', sheet_name='testset_395')
    
    # 初始化 PubMedFetcher
    fetch = PubMedFetcher()
    
    # 新增 content 欄位
    df['content'] = ''
    
    # 處理每個 PMID
    total = len(df)
    for idx, row in df.iterrows():
        pmid = str(row['PMID'])
        print(f"處理第 {idx+1}/{total} 篇文章 (PMID: {pmid})")
        
        try:
            article = fetch.article_by_pmid(pmid)
            
            # 處理標題和摘要
            title = article.title if article.title else ""
            abstract = article.abstract if article.abstract else ""
            
            # 如果標題和摘要都為空，跳過這篇文章
            if not title and not abstract:
                print(f"警告：PMID {pmid} 的標題和摘要都為空，已跳過")
                continue
                
            # 組合內容
            content = f"{title} {abstract}".strip()
            df.at[idx, 'content'] = content
            
            # 每處理 10 篇文章就儲存一次
            if (idx + 1) % 10 == 0:
                save_progress(df, idx + 1)
                
        except Exception as e:
            print(f"處理 PMID {pmid} 時發生錯誤: {str(e)}")
            continue
            
        # 加入延遲以避免過度請求
        # time.sleep(0.5)
    
    # 最終儲存
    save_final_result(df)
    print("所有文章處理完成！")

def save_progress(df, current_count):
    """儲存處理進度"""
    output_file = 'testset_395.xlsx'
    print(f"正在儲存進度... (已處理 {current_count} 篇文章)")
    # 只選擇需要的欄位
    df_to_save = df[['PMID', 'Curate (0: T0, 1: T2/4)', 'content']]
    df_to_save.to_excel(output_file, index=False)
    print(f"進度已儲存至 {output_file}")

def save_final_result(df):
    """儲存最終結果"""
    output_file = 'testset_395.xlsx'
    print("正在儲存最終結果...")
    # 只選擇需要的欄位
    df_to_save = df[['PMID', 'Curate (0: T0, 1: T2/4)', 'content']]
    df_to_save.to_excel(output_file, index=False)
    print(f"結果已儲存至 {output_file}")

if __name__ == "__main__":
    fetch_and_save_articles()