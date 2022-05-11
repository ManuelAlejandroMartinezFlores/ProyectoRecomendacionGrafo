from neo4j import GraphDatabase as GD
import pandas as pd
import numpy as np

class GenAtr:
    def __init__(self, uri, auth):
        self.driver = GD.driver(uri, auth = auth)
        
    def close(self):
        self.driver.close()
        
    
    def gen_train_data(self, df):
        c_names = ['cat_based', 'cat_cnt', 'user_based', 'user_cnt',
                   'adamic_adar', 'resource_allocation', 'link_cnt']
        df[c_names] = df.apply(self.gen_train_data_row, axis=1, result_type='expand')
        df['cat_avg'] = df['cat_based'] / df['cat_cnt']
        df['user_avg'] = df['user_based'] / df['user_cnt']
        df['adar_avg'] = df['adamic_adar'] / df['link_cnt']
        df['ra_avg'] = df['resource_allocation'] / df['link_cnt']
        df.fillna(0)
        return df 
    
    def gen_train_data_row(self, row):
        u_id, p_id, rtg = row['user_id'], row['podcast_id'], row['rating']
        self.delete_rtg(u_id, p_id)
        
        result = self.get_cat_based(u_id, p_id)
        result += self.get_user_based(u_id, p_id)
        result += self.adamic_adar(u_id, p_id)
        result += self.resource_allocation(u_id, p_id)
        
        self.create_rtg(u_id, p_id, rtg)
        
        return result
        
    def delete_rtg(self, user_id, podcast_id):
        with self.driver.session() as sess:
            sess.write_transaction(
                self._delete_rtg, user_id, podcast_id)
    
    @staticmethod
    def _delete_rtg(tx, user_id, podcast_id):
        query = (
            "MATCH (u:User)-[r]->(p:Podcast) "
            "WHERE u.id = $user_id AND p.id = $podcast_id "
            "DELETE r"
        )
        tx.run(query, user_id=user_id, podcast_id=podcast_id)

        
    def create_rtg(self, user_id, podcast_id, rating):
        with self.driver.session() as sess:
            sess.write_transaction(
                self._create_rtg, user_id, podcast_id, rating)
    
    @staticmethod
    def _create_rtg(tx, user_id, podcast_id, rating):
        query = (
            "MATCH (u:User) MATCH (p:Podcast) "
            "WHERE u.id = $user_id AND p.id = $podcast_id "
            "MERGE (u)-[r:Rating{rating:toInteger($rating)}]->(p) "
        )
        tx.run(query, user_id=user_id, podcast_id=podcast_id, rating=rating)
        
    def get_cat_based(self, user_id, podcast_id):
        with self.driver.session() as sess:
            result = sess.write_transaction(
                self._get_cat_based, user_id, podcast_id)
            return result
        
    @staticmethod
    def _get_cat_based(tx, user_id, podcast_id):
        query = (
            "MATCH (u:User)-[r]->(Podcast)-->(Category)<--(p:Podcast) "
            "WHERE u.id = $user_id AND p.id = $podcast_id "
            "RETURN r"
        )
        result = tx.run(query, user_id=user_id, podcast_id=podcast_id)
        total = 0
        cnt = 0
        for rec in result:
            total += rec['r']['rating']
            cnt += 1
        return [total, cnt]
    
    def get_user_based(self, user_id, podcast_id):
        with self.driver.session() as sess:
            result = sess.write_transaction(
                self._get_user_based, user_id, podcast_id)
            return result
        
    @staticmethod
    def _get_user_based(tx, user_id, podcast_id):
        query = (
            "MATCH (u:User)-[r1]->(Podcast)<-[r2]->(User)-[r3]->(p:Podcast) "
            "WHERE u.id = $user_id AND p.id = $podcast_id "
            "RETURN r1.rating + r2.rating + r3.rating "
            "AS total"
        )
        result = tx.run(query, user_id=user_id, podcast_id=podcast_id)
        total = 0
        cnt = 0
        for rec in result:
            total += rec['total']
            cnt += 1
        return [total, cnt]
    
    def adamic_adar(self, user_id, podcast_id):
        with self.driver.session() as sess:
            result = sess.write_transaction(
                self._adamic_adar, user_id, podcast_id)
            return result
        
    @staticmethod
    def _adamic_adar(tx, user_id, podcast_id):
        query = (
            "MATCH (u:User)-[r]->(p1:Podcast) MATCH (p:Podcast)"
            "WHERE u.id = $user_id AND p.id = $podcast_id "
            "RETURN r.rating * gds.alpha.linkprediction.adamicAdar(p1, p) "
            "AS score "
        )
        result = tx.run(query, user_id=user_id, podcast_id=podcast_id)
        total = 0
        for rec in result:
            total += rec['score']
        return [total]
    
    def resource_allocation(self, user_id, podcast_id):
        with self.driver.session() as sess:
            result = sess.write_transaction(
                self._resource_allocation, user_id, podcast_id)
            return result
        
    @staticmethod
    def _resource_allocation(tx, user_id, podcast_id):
        query = (
            "MATCH (u:User)-[r]->(p1:Podcast) MATCH (p:Podcast)"
            "WHERE u.id = $user_id AND p.id = $podcast_id "
            "RETURN r.rating * gds.alpha.linkprediction.resourceAllocation(p1, p) "
            "AS score "
        )
        result = tx.run(query, user_id=user_id, podcast_id=podcast_id)
        total = 0
        cnt = 0
        for rec in result:
            total += rec['score']
            cnt += 1
        return [total, cnt]