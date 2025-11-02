"""Tạo dữ liệu mẫu để kiểm thử RAG Agent.

Script này tạo dữ liệu mẫu trong:
1. Milvus (KB) - Các đoạn tài liệu
2. Neo4j (KG) - Các nút Project, Requirement, UserStory với quan hệ

Cách sử dụng:
    python examples/create_mock_data.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Thêm src vào đường dẫn
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from pymilvus import Collection, connections, utility

from core.config import load_config
from db.milvus_client import (
    connect_to_milvus,
    ensure_gsoft_docs_collection,
    DOC_COLLECTION_NAME,
)
from data_pipeline.embedder import VietnameseE5Embedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_mock_milvus_data(config):
    """Tạo các đoạn tài liệu mẫu trong Milvus."""
    logger.info("Đang tạo dữ liệu mẫu trong Milvus...")
    
    try:
        # Kết nối tới Milvus
        alias = connect_to_milvus(
            uri=config.milvus.uri,
            alias=config.milvus.alias,
            db_name=config.milvus.db_name,
        )
        
        # Import utility để kiểm tra/xóa collection
        from pymilvus import utility
        
        # Kiểm tra xem collection có tồn tại không và xóa nếu dimension không khớp
        collection_name = DOC_COLLECTION_NAME
        if utility.has_collection(collection_name, using=alias):
            try:
                collection = Collection(collection_name, using=alias)
                # Kiểm tra dimension của schema
                dense_vec_field = None
                for field in collection.schema.fields:
                    if field.name == "dense_vec":
                        dense_vec_field = field
                        break
                
                # Nếu dimension không khớp, xóa và tạo lại
                if dense_vec_field and dense_vec_field.params.get("dim") != config.milvus.doc_dense_dim:
                    logger.info(f"Collection tồn tại với dimension {dense_vec_field.params.get('dim')}, nhưng config yêu cầu {config.milvus.doc_dense_dim}. Đang xóa collection cũ...")
                    utility.drop_collection(collection_name, using=alias)
                    logger.info("Collection cũ đã được xóa. Sẽ tạo mới.")
                elif hasattr(collection, "num_entities"):
                    try:
                        count = collection.num_entities
                        if count > 0:
                            logger.info(f"Collection đã có {count} entities. Đang xóa collection cũ để tạo lại với dữ liệu mới...")
                            utility.drop_collection(collection_name, using=alias)
                            logger.info("Collection cũ đã được xóa. Sẽ tạo mới với dữ liệu mở rộng.")
                    except:
                        pass
            except Exception as e:
                logger.warning(f"Lỗi khi kiểm tra collection hiện có: {e}")
        
        # Đảm bảo collection tồn tại với dimension đúng
        collection = ensure_gsoft_docs_collection(alias=alias, dense_dim=config.milvus.doc_dense_dim)
        
        # Không load collection trước khi insert - có thể insert mà không cần load
        # Chỉ cần load sau khi tạo index để search
        if not hasattr(collection, "insert"):
            logger.warning("Milvus collection đang offline. Không thể chèn dữ liệu mẫu.")
            return
        
        # Khởi tạo embedder
        embedder = VietnameseE5Embedder(
            model_name=config.embedding.model_name,
            device=config.embedding.device,
            batch_size=config.embedding.batch_size,
            normalize=config.embedding.normalize_embeddings,
        )
        
        # Tài liệu mẫu - Mở rộng với nội dung chi tiết hơn
        mock_docs = []
        
        # Tài liệu Dự án Alpha
        mock_docs.extend([
            {
                "id": f"doc_{i:03d}",
                "original_doc_id": "orig_001",
                "text_preview": f"Tài liệu Dự án Alpha phần {i}: Ứng dụng web toàn diện cho quản lý quan hệ khách hàng (CRM). Bao gồm các tính năng nâng cao cho xác thực người dùng, lưu trữ dữ liệu, báo cáo và phân tích. Hệ thống hỗ trợ kiến trúc đa tenant với kiểm soát truy cập dựa trên vai trò.",
                "source": "internal",
                "url": f"https://example.com/docs/project-alpha/section-{i}",
            }
            for i in range(1, 8)
        ])
        
        # Tài liệu Dự án Beta
        mock_docs.extend([
            {
                "id": f"doc_{i:03d}",
                "original_doc_id": "orig_002",
                "text_preview": f"Tài liệu kỹ thuật Dự án Beta phần {i}: Hệ thống đồng bộ dữ liệu thời gian thực với hỗ trợ nhiều giao thức bao gồm REST API, GraphQL và WebSocket. Được thiết kế cho các kịch bản hiệu suất cao với khả năng mở rộng ngang.",
                "source": "internal",
                "url": f"https://example.com/docs/project-beta/section-{i}",
            }
            for i in range(8, 14)
        ])
        
        # Tài liệu Yêu cầu
        mock_docs.extend([
            {
                "id": "doc_014",
                "original_doc_id": "orig_003",
                "text_preview": "Yêu cầu REQ-001: Hệ thống phải cho phép người dùng đăng nhập bằng email và mật khẩu. Xác thực phải an toàn sử dụng OAuth 2.0 với hỗ trợ xác thực đa yếu tố. Quản lý phiên bao gồm tự động timeout và cơ chế refresh token.",
                "source": "srs",
                "url": "https://example.com/docs/requirements/req-001",
            },
            {
                "id": "doc_015",
                "original_doc_id": "orig_003",
                "text_preview": "Yêu cầu REQ-002: Hệ thống phải hỗ trợ kiểm soát truy cập dựa trên vai trò (RBAC) với mô hình quyền phân cấp. Người dùng phải có các quyền khác nhau dựa trên vai trò của họ bao gồm Admin, Manager, User và Guest. Kế thừa quyền phải được hỗ trợ.",
                "source": "srs",
                "url": "https://example.com/docs/requirements/req-002",
            },
            {
                "id": "doc_016",
                "original_doc_id": "orig_003",
                "text_preview": "Yêu cầu REQ-003: Người dùng phải có thể xem lịch sử đơn hàng của họ với khả năng lọc và sắp xếp nâng cao. Hệ thống phải hỗ trợ phân trang, lọc theo khoảng thời gian, lọc theo trạng thái và chức năng xuất ra định dạng CSV và PDF.",
                "source": "srs",
                "url": "https://example.com/docs/requirements/req-003",
            },
            {
                "id": "doc_017",
                "original_doc_id": "orig_003",
                "text_preview": "Yêu cầu REQ-004: Hệ thống phải cung cấp các endpoint RESTful API cho tất cả chức năng cốt lõi. API phải tuân theo đặc tả OpenAPI 3.0 với tài liệu toàn diện. Giới hạn tốc độ và xác thực qua API keys phải được triển khai.",
                "source": "srs",
                "url": "https://example.com/docs/requirements/req-004",
            },
            {
                "id": "doc_018",
                "original_doc_id": "orig_003",
                "text_preview": "Yêu cầu REQ-005: Các thao tác cơ sở dữ liệu phải được tối ưu hóa cho hiệu suất. Hệ thống phải triển khai connection pooling, tối ưu hóa truy vấn và cơ chế caching. Hỗ trợ read replicas và database sharding phải được xem xét để mở rộng.",
                "source": "technical",
                "url": "https://example.com/docs/requirements/req-005",
            },
        ])
        
        # Tài liệu User Stories
        mock_docs.extend([
            {
                "id": "doc_019",
                "original_doc_id": "orig_004",
                "text_preview": "User Story US-001: Là khách hàng, tôi muốn xem lịch sử đơn hàng của mình để tôi có thể theo dõi các giao dịch mua hàng và xem lại các giao dịch trước đó. Giao diện sẽ hiển thị đơn hàng theo danh sách thời gian với các tùy chọn tìm kiếm và lọc.",
                "source": "backlog",
                "url": "https://example.com/docs/user-stories/us-001",
            },
            {
                "id": "doc_020",
                "original_doc_id": "orig_004",
                "text_preview": "User Story US-002: Là người dùng, tôi muốn đăng nhập an toàn bằng email và mật khẩu để tài khoản của tôi được bảo vệ khỏi truy cập trái phép. Hệ thống phải hỗ trợ khôi phục mật khẩu và các tùy chọn xác thực đa yếu tố.",
                "source": "backlog",
                "url": "https://example.com/docs/user-stories/us-002",
            },
            {
                "id": "doc_021",
                "original_doc_id": "orig_004",
                "text_preview": "User Story US-003: Là admin, tôi muốn gán vai trò cho người dùng để tôi có thể kiểm soát quyền truy cập trên toàn hệ thống. Giao diện phải cung cấp quản lý vai trò với khả năng kế thừa và ghi đè quyền.",
                "source": "backlog",
                "url": "https://example.com/docs/user-stories/us-003",
            },
            {
                "id": "doc_022",
                "original_doc_id": "orig_004",
                "text_preview": "User Story US-004: Là developer, tôi muốn truy cập tài liệu API để tôi có thể tích hợp các hệ thống bên ngoài. Tài liệu phải bao gồm các mẫu mã, ví dụ xác thực và công cụ kiểm thử API tương tác.",
                "source": "backlog",
                "url": "https://example.com/docs/user-stories/us-004",
            },
        ])
        
        # Tài liệu kỹ thuật bổ sung
        mock_docs.extend([
            {
                "id": f"doc_{i:03d}",
                "original_doc_id": "orig_005",
                "text_preview": f"Tài liệu kỹ thuật phần {i}: Chi tiết kiến trúc hệ thống bao gồm thiết kế microservices, triển khai message queue, chiến lược caching và quy trình triển khai. Bao gồm khôi phục thảm họa và quy trình sao lưu.",
                "source": "internal",
                "url": f"https://example.com/docs/technical/arch-{i}",
            }
            for i in range(23, 30)
        ])
        
        # Tài liệu Dự án Gamma (dự án mới)
        mock_docs.extend([
            {
                "id": f"doc_{i:03d}",
                "original_doc_id": "orig_006",
                "text_preview": f"Tài liệu Dự án Gamma phần {i-29}: Phát triển ứng dụng di động cho nền tảng iOS và Android. Bao gồm hệ thống thông báo đẩy, hỗ trợ chế độ offline và các tính năng đồng bộ đám mây.",
                "source": "internal",
                "url": f"https://example.com/docs/project-gamma/section-{i-29}",
            }
            for i in range(30, 35)
        ])
        
        # Chuẩn bị dữ liệu
        ids = [doc["id"] for doc in mock_docs]
        original_doc_ids = [doc["original_doc_id"] for doc in mock_docs]
        permissions = [1] * len(mock_docs)  # mức quyền 1
        sources = [doc["source"] for doc in mock_docs]
        urls = [doc["url"] for doc in mock_docs]
        updated_ats = [int(datetime.now().timestamp() * 1000)] * len(mock_docs)
        text_previews = [doc["text_preview"] for doc in mock_docs]
        
        # Tạo embeddings
        logger.info("Đang tạo embeddings cho các tài liệu mẫu...")
        dense_vectors = embedder.encode_passages(text_previews)
        
        # Tạo sparse vectors (giả - dict trống cho mỗi doc)
        # Lưu ý: Sparse vectors phải là các đối tượng dict-like với {index: value}
        # Hiện tại, chúng ta sẽ sử dụng dict trống
        sparse_vectors = [{} for _ in range(len(mock_docs))]
        
        # Chèn dữ liệu
        logger.info("Đang chèn các tài liệu mẫu vào Milvus...")
        collection.insert([
            ids,
            original_doc_ids,
            permissions,
            sources,
            urls,
            updated_ats,
            text_previews,
            dense_vectors,
            sparse_vectors,
        ])
        
        # Ghi dữ liệu xuống đĩa
        logger.info("Đang ghi dữ liệu xuống đĩa...")
        collection.flush()
        
        # Tạo index nếu chưa tồn tại
        logger.info("Đang tạo indexes cho các trường vector...")
        
        # Kiểm tra và tạo index cho dense_vec
        try:
            indexes = collection.indexes
            has_dense_index = any(idx.field_name == "dense_vec" for idx in indexes)
        except:
            has_dense_index = False
        
        if not has_dense_index:
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index(
                field_name="dense_vec",
                index_params=index_params
            )
            logger.info("Index cho dense_vec đã được tạo thành công")
        else:
            logger.info("Index cho dense_vec đã tồn tại")
        
        # Tạo index cho sparse_vec (Milvus yêu cầu ngay cả khi trống)
        # Vì chúng ta đang sử dụng dict trống, tạo một index tối thiểu
        try:
            indexes = collection.indexes
            has_sparse_index = any(idx.field_name == "sparse_vec" for idx in indexes)
        except:
            has_sparse_index = False
        
        if not has_sparse_index:
            # Tạo một index tối thiểu cho sparse_vec (ngay cả khi trống)
            sparse_index_params = {
                "metric_type": "IP",
                "index_type": "SPARSE_INVERTED_INDEX",
                "params": {}
            }
            try:
                collection.create_index(
                    field_name="sparse_vec",
                    index_params=sparse_index_params
                )
                logger.info("Index cho sparse_vec đã được tạo thành công")
            except Exception as e:
                logger.warning(f"Không thể tạo index cho sparse_vec: {e}. Điều này có thể gây ra vấn đề khi load.")
        else:
            logger.info("Index cho sparse_vec đã tồn tại")
        
        logger.info(f"Đã chèn thành công {len(mock_docs)} tài liệu vào Milvus.")
        logger.info(f"  - Tài liệu từ {len(set(doc['original_doc_id'] for doc in mock_docs))} nguồn khác nhau")
        logger.info(f"  - Bao gồm tài liệu dự án, yêu cầu, user stories và tài liệu kỹ thuật")
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo dữ liệu mẫu Milvus: {e}", exc_info=True)


def create_mock_neo4j_data(config):
    """Tạo các nút Project, Requirement và UserStory mẫu trong Neo4j."""
    logger.info("Đang tạo dữ liệu mẫu trong Neo4j...")
    
    try:
        # Kết nối tới Neo4j
        neo4j_uri = config.neo4j.uri
        if neo4j_uri.startswith("http://"):
            neo4j_uri = neo4j_uri.replace("http://", "bolt://")
        elif neo4j_uri.startswith("https://"):
            neo4j_uri = neo4j_uri.replace("https://", "bolt://")
        elif not neo4j_uri.startswith("bolt://"):
            neo4j_uri = f"bolt://{neo4j_uri}"
        
        driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(config.neo4j.user, config.neo4j.password),
        )
        
        # Xác minh kết nối
        try:
            driver.verify_connectivity()
        except ServiceUnavailable:
            logger.warning("Neo4j server không khả dụng. Không thể tạo dữ liệu mẫu.")
            driver.close()
            return
        
        with driver.session() as session:
            # Xóa dữ liệu hiện có (tùy chọn - comment ra nếu muốn giữ dữ liệu hiện có)
            logger.info("Đang xóa dữ liệu mẫu hiện có...")
            session.run("MATCH (n) WHERE n:Project OR n:Requirement OR n:UserStory DETACH DELETE n")
            
            # Tạo Projects - Mở rộng với nhiều dự án hơn
            logger.info("Đang tạo Projects mẫu...")
            projects = [
                {
                    "project_id": "PROJ-001",
                    "name": "Dự án Alpha",
                    "description": "Ứng dụng web toàn diện cho quản lý quan hệ khách hàng (CRM). Các tính năng bao gồm xác thực người dùng, lưu trữ dữ liệu, báo cáo, bảng điều khiển phân tích và hỗ trợ kiến trúc đa tenant.",
                    "version": "1.5.2",
                    "status": "active",
                    "stakeholders": ["Product Owner", "Tech Lead", "QA Lead", "UX Designer", "DevOps Engineer"],
                },
                {
                    "project_id": "PROJ-002",
                    "name": "Dự án Beta",
                    "description": "Hệ thống đồng bộ dữ liệu thời gian thực với hỗ trợ nhiều giao thức (REST API, GraphQL, WebSocket). Được thiết kế cho các kịch bản hiệu suất cao với mở rộng ngang và kiến trúc phân tán.",
                    "version": "2.3.1",
                    "status": "active",
                    "stakeholders": ["Product Owner", "Architect", "Backend Lead", "SRE"],
                },
                {
                    "project_id": "PROJ-003",
                    "name": "Dự án Gamma",
                    "description": "Ứng dụng di động đa nền tảng cho iOS và Android. Các tính năng bao gồm thông báo đẩy, chế độ offline, đồng bộ đám mây và hỗ trợ xác thực sinh trắc học.",
                    "version": "1.0.0",
                    "status": "in_progress",
                    "stakeholders": ["Product Manager", "Mobile Lead", "UX Designer", "QA Engineer"],
                },
                {
                    "project_id": "PROJ-004",
                    "name": "Dự án Delta",
                    "description": "Nền tảng phân tích và báo cáo với trực quan hóa dữ liệu nâng cao, trình tạo bảng điều khiển tùy chỉnh, báo cáo theo lịch và khả năng xuất ra nhiều định dạng.",
                    "version": "0.9.0",
                    "status": "in_progress",
                    "stakeholders": ["Product Owner", "Data Analyst", "Frontend Lead", "Backend Engineer"],
                },
                {
                    "project_id": "PROJ-005",
                    "name": "Dự án Epsilon",
                    "description": "API Gateway và nền tảng điều phối microservices. Bao gồm service discovery, cân bằng tải, giới hạn tốc độ, phiên bản API và giám sát toàn diện.",
                    "version": "3.1.0",
                    "status": "active",
                    "stakeholders": ["Architect", "DevOps Lead", "Platform Engineer", "Security Specialist"],
                },
            ]
            
            for proj in projects:
                session.run(
                    """
                    CREATE (p:Project {
                        project_id: $project_id,
                        name: $name,
                        description: $description,
                        version: $version,
                        status: $status,
                        stakeholders: $stakeholders,
                        created_date: datetime(),
                        updated_date: datetime()
                    })
                    """,
                    proj
                )
            
            # Tạo Requirements - Mở rộng với các yêu cầu chi tiết hơn
            logger.info("Đang tạo Requirements mẫu...")
            requirements = [
                {
                    "req_id": "REQ-001",
                    "title": "Xác thực Người dùng",
                    "description": "Hệ thống phải cho phép người dùng đăng nhập bằng email và mật khẩu sử dụng giao thức OAuth 2.0. Hỗ trợ xác thực đa yếu tố (MFA) bao gồm SMS và ứng dụng xác thực phải được triển khai. Quản lý phiên bao gồm tự động timeout và cơ chế refresh token.",
                    "type": "functional",
                    "priority": "critical",
                    "status": "approved",
                    "version": "1.0",
                    "source": "SRS Mục 3.1",
                    "acceptance_criteria": [
                        "Người dùng có thể đăng nhập bằng email và mật khẩu hợp lệ",
                        "Người dùng nhận được thông báo lỗi rõ ràng cho thông tin đăng nhập không hợp lệ",
                        "Phiên được duy trì sau khi đăng nhập thành công với timeout có thể cấu hình",
                        "Cơ chế refresh token hoạt động chính xác",
                        "MFA có thể được bật cho mỗi tài khoản người dùng"
                    ],
                    "constraints": ["Phải sử dụng OAuth 2.0", "Mật khẩu phải được mã hóa bằng bcrypt", "Độ dài mật khẩu tối thiểu: 8 ký tự"],
                    "assumptions": ["Người dùng có địa chỉ email hợp lệ", "Người dùng có quyền truy cập SMS hoặc ứng dụng xác thực cho MFA"],
                },
                {
                    "req_id": "REQ-002",
                    "title": "Kiểm soát Truy cập Dựa trên Vai trò",
                    "description": "Hệ thống phải hỗ trợ kiểm soát truy cập dựa trên vai trò (RBAC) với mô hình quyền phân cấp. Người dùng phải có các quyền khác nhau dựa trên vai trò của họ bao gồm Admin, Manager, User và Guest. Kế thừa quyền phải được hỗ trợ với khả năng ghi đè.",
                    "type": "functional",
                    "priority": "high",
                    "status": "approved",
                    "version": "1.0",
                    "source": "SRS Mục 3.2",
                    "acceptance_criteria": [
                        "Admin có thể truy cập tất cả tính năng mà không bị hạn chế",
                        "Người dùng chỉ có thể truy cập các tính năng được gán dựa trên vai trò",
                        "Quyền được thực thi ở cả cấp UI và API",
                        "Phân cấp vai trò hoạt động chính xác với kế thừa",
                        "Ghi đè quyền được áp dụng đúng cách"
                    ],
                    "constraints": ["Phải hỗ trợ ít nhất 4 vai trò định sẵn", "Thay đổi quyền phải có hiệu lực ngay lập tức"],
                    "assumptions": ["Vai trò được định sẵn trong hệ thống", "Việc gán vai trò người dùng được quản lý bởi admin"],
                },
                {
                    "req_id": "REQ-003",
                    "title": "Xem Lịch sử Đơn hàng",
                    "description": "Người dùng phải có thể xem lịch sử đơn hàng của họ với khả năng lọc và sắp xếp nâng cao. Hệ thống phải hỗ trợ phân trang, lọc theo khoảng thời gian, lọc theo trạng thái và chức năng xuất ra định dạng CSV và PDF.",
                    "type": "functional",
                    "priority": "medium",
                    "status": "approved",
                    "version": "1.0",
                    "source": "SRS Mục 4.1",
                    "acceptance_criteria": [
                        "Người dùng có thể thấy danh sách phân trang các đơn hàng trước đó",
                        "Đơn hàng được sắp xếp theo ngày (mới nhất trước) theo mặc định",
                        "Người dùng có thể lọc đơn hàng theo trạng thái (pending, completed, cancelled)",
                        "Người dùng có thể lọc đơn hàng theo khoảng thời gian",
                        "Người dùng có thể xuất danh sách đơn hàng ra định dạng CSV",
                        "Người dùng có thể xuất chi tiết đơn hàng ra định dạng PDF"
                    ],
                    "constraints": ["Tối đa 100 đơn hàng mỗi trang", "Kích thước file xuất giới hạn 10MB"],
                    "assumptions": ["Đơn hàng được lưu trong database", "Người dùng có quyền xem đơn hàng"],
                },
                {
                    "req_id": "REQ-004",
                    "title": "RESTful API Endpoints",
                    "description": "Hệ thống phải cung cấp các endpoint RESTful API cho tất cả chức năng cốt lõi. API phải tuân theo đặc tả OpenAPI 3.0 với tài liệu toàn diện. Giới hạn tốc độ và xác thực qua API keys phải được triển khai.",
                    "type": "technical",
                    "priority": "high",
                    "status": "approved",
                    "version": "1.0",
                    "source": "SRS Mục 5.1",
                    "acceptance_criteria": [
                        "Tất cả thao tác CRUD có endpoint API tương ứng",
                        "Tài liệu API tuân theo đặc tả OpenAPI 3.0",
                        "Giới hạn tốc độ được thực thi cho mỗi API key",
                        "Xác thực API qua bearer token hoạt động chính xác",
                        "Phản hồi lỗi tuân theo mã trạng thái HTTP tiêu chuẩn"
                    ],
                    "constraints": ["Tối đa 1000 yêu cầu mỗi phút mỗi API key", "Thời gian phản hồi API phải dưới 200ms"],
                    "assumptions": ["Người tiêu dùng API có API keys hợp lệ", "Độ trễ mạng trong phạm vi chấp nhận được"],
                },
                {
                    "req_id": "REQ-005",
                    "title": "Tối ưu hóa Hiệu suất Database",
                    "description": "Các thao tác database phải được tối ưu hóa cho hiệu suất. Hệ thống phải triển khai connection pooling, tối ưu hóa truy vấn và cơ chế caching. Hỗ trợ read replicas và database sharding phải được xem xét để mở rộng.",
                    "type": "non-functional",
                    "priority": "high",
                    "status": "approved",
                    "version": "1.0",
                    "source": "SRS Mục 6.2",
                    "acceptance_criteria": [
                        "Kích thước connection pool có thể cấu hình và được tối ưu hóa",
                        "Các truy vấn database được tối ưu hóa với indexing phù hợp",
                        "Lớp caching giảm tải database ít nhất 50%",
                        "Read replicas xử lý các truy vấn đọc thành công",
                        "Chiến lược database sharding được triển khai"
                    ],
                    "constraints": ["Kích thước connection pool tối đa: 100", "Cache TTL: 5 phút"],
                    "assumptions": ["Database hỗ trợ read replicas", "Redis có sẵn để caching"],
                },
                {
                    "req_id": "REQ-006",
                    "title": "Hệ thống Thông báo Đẩy",
                    "description": "Hệ thống phải hỗ trợ thông báo đẩy cho ứng dụng di động trên cả nền tảng iOS và Android. Giao hàng thông báo phải đáng tin cậy với cơ chế thử lại và theo dõi trạng thái giao hàng.",
                    "type": "functional",
                    "priority": "medium",
                    "status": "in_progress",
                    "version": "1.0",
                    "source": "SRS Mục 7.1",
                    "acceptance_criteria": [
                        "Thông báo đẩy được gửi đến thiết bị iOS",
                        "Thông báo đẩy được gửi đến thiết bị Android",
                        "Trạng thái giao hàng thông báo được theo dõi",
                        "Thông báo thất bại được thử lại tự động",
                        "Người dùng có thể cấu hình tùy chọn thông báo"
                    ],
                    "constraints": ["Phải hỗ trợ Firebase Cloud Messaging", "Số lần thử lại tối đa: 3"],
                    "assumptions": ["Ứng dụng di động được cấu hình đúng", "Dự án Firebase đã được thiết lập"],
                },
                {
                    "req_id": "REQ-007",
                    "title": "Hỗ trợ Chế độ Offline",
                    "description": "Ứng dụng di động phải hỗ trợ chế độ offline với lưu trữ dữ liệu cục bộ. Các thay đổi được thực hiện khi offline sẽ được đồng bộ khi kết nối được khôi phục. Chiến lược giải quyết xung đột phải được triển khai.",
                    "type": "functional",
                    "priority": "medium",
                    "status": "draft",
                    "version": "1.0",
                    "source": "SRS Mục 7.2",
                    "acceptance_criteria": [
                        "Ứng dụng hoạt động chính xác ở chế độ offline",
                        "Thay đổi dữ liệu được lưu cục bộ khi offline",
                        "Đồng bộ tự động xảy ra khi kết nối được khôi phục",
                        "Giải quyết xung đột xử lý các chỉnh sửa đồng thời chính xác",
                        "Người dùng được thông báo trạng thái đồng bộ"
                    ],
                    "constraints": ["Lưu trữ cục bộ tối đa: 100MB", "Đồng bộ phải hoàn thành trong vòng 5 phút"],
                    "assumptions": ["Thiết bị có đủ dung lượng lưu trữ", "Kết nối mạng cuối cùng sẽ được khôi phục"],
                },
                {
                    "req_id": "REQ-008",
                    "title": "Bảng điều khiển Phân tích",
                    "description": "Hệ thống phải cung cấp bảng điều khiển phân tích với số liệu thời gian thực, biểu đồ trực quan hóa dữ liệu và trình tạo báo cáo tùy chỉnh. Bảng điều khiển phải hỗ trợ nhiều loại biểu đồ và khả năng xuất.",
                    "type": "functional",
                    "priority": "high",
                    "status": "in_progress",
                    "version": "1.0",
                    "source": "SRS Mục 8.1",
                    "acceptance_criteria": [
                        "Bảng điều khiển hiển thị số liệu thời gian thực với tự động làm mới",
                        "Nhiều loại biểu đồ được hỗ trợ (đường, cột, tròn, v.v.)",
                        "Trình tạo báo cáo tùy chỉnh cho phép widget kéo và thả",
                        "Báo cáo có thể được xuất ra PDF và Excel",
                        "Hiệu suất bảng điều khiển được tối ưu hóa cho tập dữ liệu lớn"
                    ],
                    "constraints": ["Số lượng widget tối đa mỗi bảng điều khiển: 20", "Khoảng thời gian làm mới dữ liệu: tối thiểu 30 giây"],
                    "assumptions": ["Dữ liệu phân tích có sẵn", "Người dùng có quyền truy cập trình duyệt hiện đại"],
                },
                {
                    "req_id": "REQ-009",
                    "title": "Tích hợp API Gateway",
                    "description": "Tất cả yêu cầu API phải được định tuyến qua API Gateway với hỗ trợ service discovery, cân bằng tải, giới hạn tốc độ, phiên bản API và giám sát toàn diện.",
                    "type": "technical",
                    "priority": "critical",
                    "status": "approved",
                    "version": "1.0",
                    "source": "SRS Mục 9.1",
                    "acceptance_criteria": [
                        "API Gateway định tuyến yêu cầu đến microservices đúng",
                        "Cân bằng tải phân phối lưu lượng đều",
                        "Giới hạn tốc độ ngăn chặn lạm dụng",
                        "Phiên bản API cho phép tương thích ngược",
                        "Bảng điều khiển giám sát hiển thị số liệu thời gian thực"
                    ],
                    "constraints": ["Giới hạn tốc độ: 100 yêu cầu/giây mỗi người dùng", "Thời gian phản hồi Gateway: dưới 50ms"],
                    "assumptions": ["Microservices được đăng ký trong service discovery", "Công cụ giám sát được cấu hình"],
                },
                {
                    "req_id": "REQ-010",
                    "title": "Xác thực Sinh trắc học",
                    "description": "Ứng dụng di động phải hỗ trợ xác thực sinh trắc học sử dụng vân tay hoặc nhận dạng khuôn mặt. Xác thực sinh trắc học nên là tùy chọn và có thể được bật theo sở thích người dùng.",
                    "type": "functional",
                    "priority": "medium",
                    "status": "draft",
                    "version": "1.0",
                    "source": "SRS Mục 10.1",
                    "acceptance_criteria": [
                        "Xác thực sinh trắc học hoạt động trên thiết bị iOS",
                        "Xác thực sinh trắc học hoạt động trên thiết bị Android",
                        "Người dùng có thể bật/tắt xác thực sinh trắc học trong cài đặt",
                        "Dự phòng mật khẩu có sẵn nếu sinh trắc học thất bại",
                        "Dữ liệu sinh trắc học được lưu trữ an toàn trên thiết bị"
                    ],
                    "constraints": ["Phải sử dụng API sinh trắc học gốc của thiết bị", "Giới hạn thất bại sinh trắc học: 5 lần thử"],
                    "assumptions": ["Thiết bị hỗ trợ xác thực sinh trắc học", "Người dùng đã đăng ký dữ liệu sinh trắc học"],
                },
            ]
            
            for req in requirements:
                session.run(
                    """
                    CREATE (r:Requirement {
                        req_id: $req_id,
                        title: $title,
                        description: $description,
                        type: $type,
                        priority: $priority,
                        status: $status,
                        version: $version,
                        source: $source,
                        acceptance_criteria: $acceptance_criteria,
                        constraints: $constraints,
                        assumptions: $assumptions,
                        created_date: datetime(),
                        updated_date: datetime()
                    })
                    """,
                    req
                )
            
            # Tạo User Stories - Mở rộng với các câu chuyện chi tiết hơn
            logger.info("Đang tạo User Stories mẫu...")
            user_stories = [
                {
                    "story_id": "US-001",
                    "title": "Xem Lịch sử Đơn hàng",
                    "as_a": "khách hàng",
                    "i_want": "xem lịch sử đơn hàng của mình với lọc và tìm kiếm",
                    "so_that": "tôi có thể theo dõi các giao dịch mua hàng và xem lại các giao dịch trước đó",
                    "description": "Là khách hàng, tôi muốn xem lịch sử đơn hàng của mình với khả năng lọc và tìm kiếm nâng cao để tôi có thể theo dõi các giao dịch mua hàng và xem lại các giao dịch trước đó một cách hiệu quả.",
                    "priority": "high",
                    "story_points": 5,
                    "status": "done",
                    "sprint": "Sprint 1",
                    "acceptance_criteria": [
                        "Tôi có thể thấy danh sách phân trang các đơn hàng của mình",
                        "Đơn hàng được hiển thị với ngày, tổng số tiền và trạng thái",
                        "Tôi có thể lọc đơn hàng theo khoảng thời gian",
                        "Tôi có thể lọc đơn hàng theo trạng thái",
                        "Tôi có thể tìm kiếm đơn hàng theo số đơn hàng",
                        "Tôi có thể nhấp vào một đơn hàng để xem thông tin chi tiết"
                    ],
                },
                {
                    "story_id": "US-002",
                    "title": "Đăng nhập An toàn với MFA",
                    "as_a": "người dùng",
                    "i_want": "đăng nhập an toàn với xác thực đa yếu tố",
                    "so_that": "tài khoản của tôi được bảo vệ khỏi truy cập trái phép",
                    "description": "Là người dùng, tôi muốn đăng nhập an toàn bằng email và mật khẩu với xác thực đa yếu tố tùy chọn để tài khoản của tôi được bảo vệ khỏi truy cập trái phép.",
                    "priority": "critical",
                    "story_points": 3,
                    "status": "in_progress",
                    "sprint": "Sprint 1",
                    "acceptance_criteria": [
                        "Tôi có thể đăng nhập bằng email và mật khẩu",
                        "Tôi có thể bật MFA trong cài đặt tài khoản",
                        "MFA có thể sử dụng SMS hoặc ứng dụng xác thực",
                        "Các lần thử đăng nhập thất bại được ghi log và giới hạn",
                        "Phiên hết hạn sau khoảng thời gian không hoạt động có thể cấu hình"
                    ],
                },
                {
                    "story_id": "US-003",
                    "title": "Quản lý Vai trò và Quyền Người dùng",
                    "as_a": "admin",
                    "i_want": "gán vai trò và quyền cho người dùng",
                    "so_that": "tôi có thể kiểm soát quyền truy cập trên toàn hệ thống",
                    "description": "Là admin, tôi muốn gán vai trò và quản lý quyền cho người dùng để tôi có thể kiểm soát quyền truy cập trên toàn hệ thống một cách hiệu quả.",
                    "priority": "high",
                    "story_points": 8,
                    "status": "backlog",
                    "sprint": "Sprint 2",
                    "acceptance_criteria": [
                        "Tôi có thể gán các vai trò định sẵn cho người dùng",
                        "Tôi có thể thấy danh sách tất cả các vai trò có sẵn",
                        "Tôi có thể xem quyền cho mỗi vai trò",
                        "Tôi có thể ghi đè quyền cho người dùng cụ thể",
                        "Thay đổi quyền có hiệu lực ngay lập tức"
                    ],
                },
                {
                    "story_id": "US-004",
                    "title": "Truy cập Tài liệu API",
                    "as_a": "developer",
                    "i_want": "truy cập tài liệu API toàn diện",
                    "so_that": "tôi có thể tích hợp các hệ thống bên ngoài với nền tảng",
                    "description": "Là developer, tôi muốn truy cập tài liệu API toàn diện với các ví dụ mã và công cụ kiểm thử tương tác để tôi có thể tích hợp các hệ thống bên ngoài với nền tảng một cách hiệu quả.",
                    "priority": "medium",
                    "story_points": 3,
                    "status": "backlog",
                    "sprint": "Sprint 2",
                    "acceptance_criteria": [
                        "Tài liệu API tuân theo đặc tả OpenAPI 3.0",
                        "Ví dụ mã được cung cấp cho tất cả các endpoint",
                        "Công cụ kiểm thử API tương tác có sẵn",
                        "Ví dụ xác thực được bao gồm",
                        "Ví dụ phản hồi lỗi được tài liệu hóa"
                    ],
                },
                {
                    "story_id": "US-005",
                    "title": "Nhận Thông báo Đẩy",
                    "as_a": "người dùng di động",
                    "i_want": "nhận thông báo đẩy",
                    "so_that": "tôi được thông báo về các cập nhật quan trọng",
                    "description": "Là người dùng di động, tôi muốn nhận thông báo đẩy cho các sự kiện quan trọng để tôi được thông báo về cập nhật đơn hàng, cảnh báo bảo mật và thông báo hệ thống.",
                    "priority": "medium",
                    "story_points": 5,
                    "status": "in_progress",
                    "sprint": "Sprint 3",
                    "acceptance_criteria": [
                        "Tôi nhận thông báo đẩy trên thiết bị iOS",
                        "Tôi nhận thông báo đẩy trên thiết bị Android",
                        "Tôi có thể cấu hình tùy chọn thông báo",
                        "Thông báo được giao hàng đáng tin cậy",
                        "Tôi có thể xem lịch sử thông báo"
                    ],
                },
                {
                    "story_id": "US-006",
                    "title": "Sử dụng Ứng dụng Offline",
                    "as_a": "người dùng di động",
                    "i_want": "sử dụng ứng dụng ở chế độ offline",
                    "so_that": "tôi có thể truy cập dữ liệu ngay cả khi không có kết nối internet",
                    "description": "Là người dùng di động, tôi muốn sử dụng ứng dụng ở chế độ offline với lưu trữ dữ liệu cục bộ để tôi có thể truy cập và chỉnh sửa dữ liệu ngay cả khi không có kết nối internet.",
                    "priority": "medium",
                    "story_points": 8,
                    "status": "backlog",
                    "sprint": "Sprint 3",
                    "acceptance_criteria": [
                        "Ứng dụng hoạt động chính xác không có internet",
                        "Dữ liệu được lưu cục bộ để truy cập offline",
                        "Thay đổi được tự động đồng bộ khi online",
                        "Giải quyết xung đột xử lý các chỉnh sửa đồng thời",
                        "Trạng thái đồng bộ được chỉ rõ"
                    ],
                },
                {
                    "story_id": "US-007",
                    "title": "Tạo Bảng điều khiển Phân tích Tùy chỉnh",
                    "as_a": "chuyên gia phân tích dữ liệu",
                    "i_want": "tạo bảng điều khiển phân tích tùy chỉnh",
                    "so_that": "tôi có thể trực quan hóa dữ liệu theo nhu cầu của mình",
                    "description": "Là chuyên gia phân tích dữ liệu, tôi muốn tạo bảng điều khiển phân tích tùy chỉnh với widget kéo và thả để tôi có thể trực quan hóa dữ liệu theo nhu cầu và sở thích cụ thể của mình.",
                    "priority": "high",
                    "story_points": 13,
                    "status": "in_progress",
                    "sprint": "Sprint 4",
                    "acceptance_criteria": [
                        "Tôi có thể tạo bảng điều khiển mới từ đầu",
                        "Tôi có thể thêm widget bằng giao diện kéo và thả",
                        "Nhiều loại biểu đồ có sẵn",
                        "Tôi có thể cấu hình nguồn dữ liệu widget",
                        "Bảng điều khiển có thể được lưu và chia sẻ với nhóm"
                    ],
                },
                {
                    "story_id": "US-008",
                    "title": "Xuất Báo cáo ra Nhiều Định dạng",
                    "as_a": "người dùng nghiệp vụ",
                    "i_want": "xuất báo cáo ra các định dạng khác nhau",
                    "so_that": "tôi có thể chia sẻ báo cáo với các bên liên quan",
                    "description": "Là người dùng nghiệp vụ, tôi muốn xuất báo cáo và bảng điều khiển phân tích ra nhiều định dạng (PDF, Excel, CSV) để tôi có thể chia sẻ chúng với các bên liên quan có thể sử dụng các công cụ khác nhau.",
                    "priority": "medium",
                    "story_points": 5,
                    "status": "backlog",
                    "sprint": "Sprint 4",
                    "acceptance_criteria": [
                        "Tôi có thể xuất bảng điều khiển ra định dạng PDF",
                        "Tôi có thể xuất bảng điều khiển ra định dạng Excel",
                        "Tôi có thể xuất dữ liệu ra định dạng CSV",
                        "File xuất giữ nguyên định dạng",
                        "Xuất bao gồm tất cả dữ liệu hiển thị"
                    ],
                },
                {
                    "story_id": "US-009",
                    "title": "Giám sát Hiệu suất API Gateway",
                    "as_a": "kỹ sư devops",
                    "i_want": "giám sát số liệu hiệu suất API Gateway",
                    "so_that": "tôi có thể đảm bảo độ tin cậy và hiệu suất hệ thống",
                    "description": "Là kỹ sư DevOps, tôi muốn giám sát số liệu hiệu suất API Gateway bao gồm tốc độ yêu cầu, độ trễ, tỷ lệ lỗi và tình trạng sức khỏe dịch vụ để tôi có thể đảm bảo độ tin cậy hệ thống và hiệu suất tối ưu.",
                    "priority": "critical",
                    "story_points": 8,
                    "status": "done",
                    "sprint": "Sprint 1",
                    "acceptance_criteria": [
                        "Bảng điều khiển số liệu thời gian thực có sẵn",
                        "Tốc độ yêu cầu và độ trễ được hiển thị",
                        "Tỷ lệ lỗi và mã trạng thái được theo dõi",
                        "Tình trạng sức khỏe dịch vụ được hiển thị",
                        "Cảnh báo được gửi khi vi phạm ngưỡng"
                    ],
                },
                {
                    "story_id": "US-010",
                    "title": "Bật Đăng nhập Sinh trắc học",
                    "as_a": "người dùng di động",
                    "i_want": "sử dụng xác thực sinh trắc học",
                    "so_that": "tôi có thể đăng nhập nhanh hơn và an toàn hơn",
                    "description": "Là người dùng di động, tôi muốn sử dụng xác thực sinh trắc học (vân tay hoặc nhận dạng khuôn mặt) để đăng nhập để tôi có thể truy cập tài khoản nhanh hơn và an toàn hơn mà không cần nhập mật khẩu.",
                    "priority": "medium",
                    "story_points": 5,
                    "status": "backlog",
                    "sprint": "Sprint 5",
                    "acceptance_criteria": [
                        "Đăng nhập sinh trắc học hoạt động trên thiết bị iOS",
                        "Đăng nhập sinh trắc học hoạt động trên thiết bị Android",
                        "Tôi có thể bật/tắt xác thực sinh trắc học trong cài đặt",
                        "Dự phòng mật khẩu có sẵn",
                        "Dữ liệu sinh trắc học được lưu trữ an toàn trên thiết bị"
                    ],
                },
                {
                    "story_id": "US-011",
                    "title": "Xem Số liệu Hiệu suất Database",
                    "as_a": "quản trị viên database",
                    "i_want": "xem số liệu hiệu suất database",
                    "so_that": "tôi có thể tối ưu hóa hiệu suất truy vấn và lập kế hoạch dung lượng",
                    "description": "Là quản trị viên database, tôi muốn xem số liệu hiệu suất database bao gồm thời gian thực thi truy vấn, sử dụng connection pool, tỷ lệ cache hit và độ trễ replication để tôi có thể tối ưu hóa hiệu suất và lập kế hoạch dung lượng hiệu quả.",
                    "priority": "high",
                    "story_points": 8,
                    "status": "in_progress",
                    "sprint": "Sprint 4",
                    "acceptance_criteria": [
                        "Thời gian thực thi truy vấn được hiển thị",
                        "Số liệu connection pool được hiển thị",
                        "Tỷ lệ cache hit được theo dõi",
                        "Độ trễ read replica được giám sát",
                        "Nhật ký truy vấn chậm có thể truy cập"
                    ],
                },
                {
                    "story_id": "US-012",
                    "title": "Cấu hình Quy tắc Giới hạn Tốc độ",
                    "as_a": "quản trị viên nền tảng",
                    "i_want": "cấu hình quy tắc giới hạn tốc độ API",
                    "so_that": "tôi có thể ngăn chặn lạm dụng API và đảm bảo sử dụng tài nguyên công bằng",
                    "description": "Là quản trị viên nền tảng, tôi muốn cấu hình quy tắc giới hạn tốc độ API cho mỗi người dùng, API key hoặc endpoint để tôi có thể ngăn chặn lạm dụng API và đảm bảo sử dụng tài nguyên công bằng trên tất cả người tiêu dùng.",
                    "priority": "high",
                    "story_points": 5,
                    "status": "backlog",
                    "sprint": "Sprint 2",
                    "acceptance_criteria": [
                        "Tôi có thể đặt giới hạn tốc độ cho mỗi API key",
                        "Tôi có thể đặt giới hạn tốc độ cho mỗi endpoint",
                        "Quy tắc giới hạn tốc độ có thể được cập nhật mà không cần khởi động lại",
                        "Vi phạm giới hạn tốc độ được ghi log",
                        "Người dùng nhận được thông báo lỗi rõ ràng khi bị giới hạn tốc độ"
                    ],
                },
            ]
            
            for us in user_stories:
                session.run(
                    """
                    CREATE (us:UserStory {
                        story_id: $story_id,
                        title: $title,
                        as_a: $as_a,
                        i_want: $i_want,
                        so_that: $so_that,
                        description: $description,
                        priority: $priority,
                        story_points: $story_points,
                        status: $status,
                        sprint: $sprint,
                        acceptance_criteria: $acceptance_criteria,
                        created_date: datetime(),
                        updated_date: datetime()
                    })
                    """,
                    us
                )
            
            # Tạo Quan hệ
            logger.info("Đang tạo Quan hệ mẫu...")
            
            # Project -> Requirement
            session.run(
                """
                MATCH (p:Project {project_id: 'PROJ-001'})
                MATCH (r:Requirement {req_id: 'REQ-001'})
                CREATE (p)-[:CONTAINS_REQUIREMENT]->(r)
                """
            )
            session.run(
                """
                MATCH (p:Project {project_id: 'PROJ-001'})
                MATCH (r:Requirement {req_id: 'REQ-002'})
                CREATE (p)-[:CONTAINS_REQUIREMENT]->(r)
                """
            )
            session.run(
                """
                MATCH (p:Project {project_id: 'PROJ-001'})
                MATCH (r:Requirement {req_id: 'REQ-003'})
                CREATE (p)-[:CONTAINS_REQUIREMENT]->(r)
                """
            )
            
            # Project -> UserStory
            session.run(
                """
                MATCH (p:Project {project_id: 'PROJ-001'})
                MATCH (us:UserStory {story_id: 'US-001'})
                CREATE (p)-[:CONTAINS_STORY]->(us)
                """
            )
            session.run(
                """
                MATCH (p:Project {project_id: 'PROJ-001'})
                MATCH (us:UserStory {story_id: 'US-002'})
                CREATE (p)-[:CONTAINS_STORY]->(us)
                """
            )
            session.run(
                """
                MATCH (p:Project {project_id: 'PROJ-001'})
                MATCH (us:UserStory {story_id: 'US-003'})
                CREATE (p)-[:CONTAINS_STORY]->(us)
                """
            )
            
            # UserStory -> Requirement (DERIVED_FROM)
            session.run(
                """
                MATCH (us:UserStory {story_id: 'US-001'})
                MATCH (r:Requirement {req_id: 'REQ-003'})
                CREATE (r)-[:DERIVED_FROM]->(us)
                """
            )
            session.run(
                """
                MATCH (us:UserStory {story_id: 'US-002'})
                MATCH (r:Requirement {req_id: 'REQ-001'})
                CREATE (r)-[:DERIVED_FROM]->(us)
                """
            )
            
            # Requirement -> Requirement (DEPENDS_ON)
            session.run(
                """
                MATCH (r1:Requirement {req_id: 'REQ-002'})
                MATCH (r2:Requirement {req_id: 'REQ-001'})
                CREATE (r2)-[:DEPENDS_ON]->(r1)
                """
            )
            
            # UserStory -> UserStory (RELATES_TO)
            session.run(
                """
                MATCH (us1:UserStory {story_id: 'US-001'})
                MATCH (us2:UserStory {story_id: 'US-003'})
                CREATE (us1)-[:RELATES_TO]->(us2)
                """
            )
            
            # Tạo các quan hệ phức tạp hơn
            logger.info("Đang tạo các quan hệ mở rộng...")
            
            # Project -> Requirement (nhiều kết nối hơn)
            project_req_relations = [
                ("PROJ-001", ["REQ-001", "REQ-002", "REQ-003", "REQ-004", "REQ-005"]),
                ("PROJ-002", ["REQ-004", "REQ-005", "REQ-009"]),
                ("PROJ-003", ["REQ-006", "REQ-007", "REQ-010"]),
                ("PROJ-004", ["REQ-005", "REQ-008"]),
                ("PROJ-005", ["REQ-004", "REQ-009"]),
            ]
            
            for proj_id, req_ids in project_req_relations:
                for req_id in req_ids:
                    session.run(
                        f"""
                        MATCH (p:Project {{project_id: '{proj_id}'}})
                        MATCH (r:Requirement {{req_id: '{req_id}'}})
                        MERGE (p)-[:CONTAINS_REQUIREMENT]->(r)
                        """
                    )
            
            # Project -> UserStory (nhiều kết nối hơn)
            project_story_relations = [
                ("PROJ-001", ["US-001", "US-002", "US-003", "US-004"]),
                ("PROJ-002", ["US-004", "US-009", "US-012"]),
                ("PROJ-003", ["US-005", "US-006", "US-010"]),
                ("PROJ-004", ["US-007", "US-008", "US-011"]),
                ("PROJ-005", ["US-009", "US-012"]),
            ]
            
            for proj_id, story_ids in project_story_relations:
                for story_id in story_ids:
                    session.run(
                        f"""
                        MATCH (p:Project {{project_id: '{proj_id}'}})
                        MATCH (us:UserStory {{story_id: '{story_id}'}})
                        MERGE (p)-[:CONTAINS_STORY]->(us)
                        """
                    )
            
            # UserStory -> Requirement (DERIVED_FROM)
            story_req_relations = [
                ("US-001", "REQ-003"),
                ("US-002", "REQ-001"),
                ("US-003", "REQ-002"),
                ("US-004", "REQ-004"),
                ("US-005", "REQ-006"),
                ("US-006", "REQ-007"),
                ("US-007", "REQ-008"),
                ("US-009", "REQ-009"),
                ("US-010", "REQ-010"),
                ("US-011", "REQ-005"),
                ("US-012", "REQ-009"),
            ]
            
            for story_id, req_id in story_req_relations:
                session.run(
                    f"""
                    MATCH (us:UserStory {{story_id: '{story_id}'}})
                    MATCH (r:Requirement {{req_id: '{req_id}'}})
                    MERGE (r)-[:DERIVED_FROM]->(us)
                    """
                )
            
            # Requirement -> Requirement (DEPENDS_ON)
            req_dependencies = [
                ("REQ-002", "REQ-001"),  # RBAC phụ thuộc vào Xác thực
                ("REQ-004", "REQ-001"),  # API phụ thuộc vào Xác thực
                ("REQ-008", "REQ-005"),  # Phân tích phụ thuộc vào Tối ưu DB
                ("REQ-009", "REQ-004"),  # API Gateway phụ thuộc vào API endpoints
                ("REQ-006", "REQ-001"),  # Thông báo đẩy phụ thuộc vào Xác thực
            ]
            
            for req1_id, req2_id in req_dependencies:
                session.run(
                    f"""
                    MATCH (r1:Requirement {{req_id: '{req1_id}'}})
                    MATCH (r2:Requirement {{req_id: '{req2_id}'}})
                    MERGE (r1)-[:DEPENDS_ON]->(r2)
                    """
                )
            
            # Requirement -> Requirement (REFINES)
            req_refines = [
                ("REQ-001", "REQ-010"),  # Xác thực làm rõ Xác thực Sinh trắc học
            ]
            
            for req1_id, req2_id in req_refines:
                session.run(
                    f"""
                    MATCH (r1:Requirement {{req_id: '{req1_id}'}})
                    MATCH (r2:Requirement {{req_id: '{req2_id}'}})
                    MERGE (r1)-[:REFINES]->(r2)
                    """
                )
            
            # UserStory -> UserStory (RELATES_TO)
            story_relations = [
                ("US-001", "US-002"),  # Lịch sử đơn hàng liên quan đến đăng nhập
                ("US-002", "US-003"),  # Đăng nhập liên quan đến quản lý vai trò
                ("US-005", "US-006"),  # Thông báo đẩy liên quan đến chế độ offline
                ("US-007", "US-008"),  # Bảng điều khiển liên quan đến xuất
                ("US-009", "US-012"),  # Giám sát API Gateway liên quan đến giới hạn tốc độ
            ]
            
            for story1_id, story2_id in story_relations:
                session.run(
                    f"""
                    MATCH (us1:UserStory {{story_id: '{story1_id}'}})
                    MATCH (us2:UserStory {{story_id: '{story2_id}'}})
                    MERGE (us1)-[:RELATES_TO]->(us2)
                    """
                )
            
            # UserStory -> UserStory (BLOCKS)
            story_blocks = [
                ("US-002", "US-003"),  # Đăng nhập chặn quản lý vai trò (phải đăng nhập trước)
            ]
            
            for blocker_id, blocked_id in story_blocks:
                session.run(
                    f"""
                    MATCH (us1:UserStory {{story_id: '{blocker_id}'}})
                    MATCH (us2:UserStory {{story_id: '{blocked_id}'}})
                    MERGE (us1)-[:BLOCKS]->(us2)
                    """
                )
            
            logger.info("Đã tạo thành công dữ liệu mẫu trong Neo4j!")
            logger.info("Đã tạo:")
            logger.info(f"  - {len(projects)} Projects")
            logger.info(f"  - {len(requirements)} Requirements")
            logger.info(f"  - {len(user_stories)} User Stories")
            logger.info("  - Nhiều quan hệ (CONTAINS, DERIVED_FROM, DEPENDS_ON, REFINES, RELATES_TO, BLOCKS)")
        
        driver.close()
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo dữ liệu mẫu Neo4j: {e}", exc_info=True)


def main():
    """Hàm chính để tạo dữ liệu mẫu."""
    logger.info("=" * 80)
    logger.info("Tạo Dữ liệu Mẫu cho RAG Agent")
    logger.info("=" * 80)
    
    config = load_config()
    
    # Tạo dữ liệu Milvus
    create_mock_milvus_data(config)
    
    # Tạo dữ liệu Neo4j
    create_mock_neo4j_data(config)
    
    logger.info("=" * 80)
    logger.info("Hoàn thành tạo dữ liệu mẫu!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
