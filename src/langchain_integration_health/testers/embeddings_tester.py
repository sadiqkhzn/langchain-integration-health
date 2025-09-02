from typing import Dict, Any, List
import numpy as np
from .base_tester import BaseIntegrationTester

class EmbeddingsTester(BaseIntegrationTester):
    """Specialized tester for Embeddings integrations"""
    
    REQUIRED_METHODS = [
        'embed_documents', 'embed_query'
    ]
    
    OPTIONAL_METHODS = [
        'aembed_documents', 'aembed_query'
    ]
    
    async def _test_method_functionality(self) -> Dict[str, Any]:
        """Test actual method execution for Embeddings"""
        functionality_results = {}
        
        try:
            instance = self.integration_class(**self.config)
            
            # Test document embedding
            functionality_results['embed_documents'] = await self._test_embed_documents(instance)
            
            # Test query embedding
            functionality_results['embed_query'] = await self._test_embed_query(instance)
            
            # Test async embeddings if available
            functionality_results['async_embeddings'] = await self._test_async_embeddings(instance)
            
            # Test embedding dimensions consistency
            functionality_results['dimension_consistency'] = await self._test_dimension_consistency(instance)
            
        except Exception as e:
            self.results.errors.append(f"Embeddings functionality testing failed: {str(e)}")
            
        return functionality_results
    
    async def _test_embed_documents(self, instance: Any) -> bool:
        """Test document embedding functionality"""
        try:
            test_documents = [
                "This is a test document.",
                "Another test document for embedding.",
                "A third document to test batch embedding."
            ]
            
            embeddings = instance.embed_documents(test_documents)
            
            if embeddings and len(embeddings) == len(test_documents):
                # Check if embeddings are valid vectors
                if all(isinstance(emb, list) and len(emb) > 0 for emb in embeddings):
                    self.results.performance_metrics['documents_embedded'] = len(test_documents)
                    return True
                else:
                    self.results.errors.append("embed_documents returned invalid embedding format")
                    return False
            else:
                self.results.errors.append("embed_documents returned wrong number of embeddings")
                return False
                
        except Exception as e:
            self.results.errors.append(f"embed_documents test failed: {str(e)}")
            return False
    
    async def _test_embed_query(self, instance: Any) -> bool:
        """Test query embedding functionality"""
        try:
            test_query = "What is the meaning of life?"
            
            embedding = instance.embed_query(test_query)
            
            if embedding and isinstance(embedding, list) and len(embedding) > 0:
                self.results.performance_metrics['embedding_dimension'] = len(embedding)
                return True
            else:
                self.results.errors.append("embed_query returned invalid embedding")
                return False
                
        except Exception as e:
            self.results.errors.append(f"embed_query test failed: {str(e)}")
            return False
    
    async def _test_async_embeddings(self, instance: Any) -> bool:
        """Test async embedding capabilities"""
        try:
            if hasattr(instance, 'aembed_query'):
                test_query = "Async embedding test query"
                embedding = await instance.aembed_query(test_query)
                
                if embedding and isinstance(embedding, list) and len(embedding) > 0:
                    self.results.async_support = True
                    return True
                    
            if hasattr(instance, 'aembed_documents'):
                test_docs = ["Async document embedding test"]
                embeddings = await instance.aembed_documents(test_docs)
                
                if embeddings and len(embeddings) == 1:
                    self.results.async_support = True
                    return True
                    
            self.results.warnings.append("No async embedding methods available")
            return False
            
        except Exception as e:
            self.results.warnings.append(f"Async embeddings test failed: {str(e)}")
            return False
    
    async def _test_dimension_consistency(self, instance: Any) -> bool:
        """Test that embeddings have consistent dimensions"""
        try:
            # Test multiple queries to ensure consistent dimensions
            test_queries = [
                "Short query",
                "This is a much longer query with more words to test dimension consistency",
                "Medium length query for testing"
            ]
            
            dimensions = []
            for query in test_queries:
                embedding = instance.embed_query(query)
                if embedding and isinstance(embedding, list):
                    dimensions.append(len(embedding))
                    
            if dimensions and len(set(dimensions)) == 1:
                # All dimensions are the same
                return True
            else:
                self.results.errors.append(f"Inconsistent embedding dimensions: {dimensions}")
                return False
                
        except Exception as e:
            self.results.warnings.append(f"Dimension consistency test failed: {str(e)}")
            return False
    
    async def _test_specific_error_scenarios(self, instance: Any) -> Dict[str, bool]:
        """Test embeddings-specific error scenarios"""
        error_scenarios = {}
        
        # Test empty document list
        try:
            instance.embed_documents([])
            error_scenarios['handles_empty_documents'] = True
        except Exception:
            error_scenarios['handles_empty_documents'] = False
            
        # Test empty query
        try:
            instance.embed_query("")
            error_scenarios['handles_empty_query'] = True
        except Exception:
            error_scenarios['handles_empty_query'] = False
            
        # Test very large document
        try:
            large_doc = "word " * 50000
            instance.embed_query(large_doc)
            error_scenarios['handles_large_input'] = True
        except Exception:
            error_scenarios['handles_large_input'] = False
            
        return error_scenarios