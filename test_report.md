# 麦当劳用户行为预测系统测试报告

## 项目概述
这是一个基于决策树的用户行为预测项目，用于预测用户是否喜欢麦当劳。项目包含核心预测模型和Flask API接口。

---

## 1. 测试方案

| 测试类型 | 测试目标 | 测试范围 | 技术栈 |
|---------|---------|---------|--------|
| 单元测试 | 验证核心预测模型的各个功能模块 | McDonaldPredictor类的所有方法 | pytest |
| API测试 | 验证HTTP接口的正确性和错误处理 | Flask应用的所有路由 | pytest + Flask测试客户端 |
| 集成测试 | 验证系统各组件间的交互 | 模型初始化→训练→预测的完整流程 | pytest |

---

## 2. 测试用例设计

### 2.1 单元测试用例 (test_mcdonald_predictor.py)

| 测试ID | 测试用例名称 | 测试描述 | 预期结果 |
|-------|-------------|---------|---------|
| UT-001 | test_init_model_loading | 测试模型初始化和加载功能 | 模型应成功加载或训练，文件应存在 |
| UT-002 | test_preprocess | 测试数据预处理功能 | 应过滤掉18岁以下用户，保留有效数据 |
| UT-003 | test_train_model | 测试模型训练功能 | 模型应包含scaler和classifier组件 |
| UT-004 | test_predict_single | 测试单次预测功能 | 返回结果应包含prediction、confidence、feature_importance |
| UT-005 | test_predict_single_different_genders | 测试不同性别输入的预测 | 男性和女性输入都应得到有效预测 |
| UT-006 | test_predict_single_different_visit_frequencies | 测试不同访问频率的预测 | 所有访问频率(rarely/monthly/weekly/daily)都应得到有效预测 |
| UT-007 | test_feature_importance_sum | 测试特征重要性总和 | 所有特征重要性之和应接近1 |
| UT-008 | test_predict_single_edge_case_age | 测试边缘年龄输入 | 18岁用户应得到有效预测 |
| UT-009 | test_predict_single_high_income | 测试高收入用户预测 | 高收入用户应得到有效预测 |

### 2.2 API测试用例 (test_api.py)

| 测试ID | 测试用例名称 | 测试描述 | 预期结果 |
|-------|-------------|---------|---------|
| API-001 | test_index_route | 测试首页路由 | 返回状态码200 |
| API-002 | test_predict_route_valid_input | 测试有效输入的预测 | 返回状态码200，包含完整预测结果 |
| API-003 | test_predict_route_invalid_content_type | 测试无效内容类型 | 返回状态码400/415/500 |
| API-004 | test_predict_route_missing_fields | 测试缺失字段的请求 | 返回状态码400，包含错误信息 |
| API-005 | test_predict_route_empty_body | 测试空请求体 | 返回状态码400，包含错误信息 |
| API-006 | test_predict_route_invalid_http_method | 测试无效HTTP方法 | 返回状态码405 |
| API-007 | test_predict_route_different_categories | 测试不同分类值的预测 | 所有分类组合都应得到有效预测 |

### 2.3 集成测试用例 (test_integration.py)

| 测试ID | 测试用例名称 | 测试描述 | 预期结果 |
|-------|-------------|---------|---------|
| INT-001 | test_end_to_end_prediction | 测试端到端预测流程 | 从模型初始化到预测应完整执行 |
| INT-002 | test_api_integration | 测试API与模型的集成 | 首次请求应训练模型，后续请求应使用加载的模型 |
| INT-003 | test_data_consistency | 测试模型与训练数据的一致性 | 训练数据样本的预测应具有高置信度 |
| INT-004 | test_multiple_predictions | 测试多次连续预测 | 多个不同输入应得到一致且有效的预测 |
| INT-005 | test_api_error_handling_integration | 测试API错误处理集成 | 各种错误场景应返回适当的HTTP状态码 |

---

## 3. 测试执行情况

### 3.1 单元测试结果
| 测试ID | 测试用例名称 | 执行结果 |
|-------|-------------|---------|
| UT-001 | test_init_model_loading | ✅ PASS |
| UT-002 | test_preprocess | ✅ PASS |
| UT-003 | test_train_model | ✅ PASS |
| UT-004 | test_predict_single | ✅ PASS |
| UT-005 | test_predict_single_different_genders | ✅ PASS |
| UT-006 | test_predict_single_different_visit_frequencies | ✅ PASS |
| UT-007 | test_feature_importance_sum | ✅ PASS |
| UT-008 | test_predict_single_edge_case_age | ✅ PASS |
| UT-009 | test_predict_single_high_income | ✅ PASS |

### 3.2 API测试结果
| 测试ID | 测试用例名称 | 执行结果 |
|-------|-------------|---------|
| API-001 | test_index_route | ✅ PASS |
| API-002 | test_predict_route_valid_input | ✅ PASS |
| API-003 | test_predict_route_invalid_content_type | ✅ PASS |
| API-004 | test_predict_route_missing_fields | ✅ PASS |
| API-005 | test_predict_route_empty_body | ✅ PASS |
| API-006 | test_predict_route_invalid_http_method | ✅ PASS |
| API-007 | test_predict_route_different_categories | ✅ PASS |

### 3.3 集成测试结果
| 测试ID | 测试用例名称 | 执行结果 |
|-------|-------------|---------|
| INT-001 | test_end_to_end_prediction | ✅ PASS |
| INT-002 | test_api_integration | ✅ PASS |
| INT-003 | test_data_consistency | ✅ PASS |
| INT-004 | test_multiple_predictions | ✅ PASS |
| INT-005 | test_api_error_handling_integration | ✅ PASS |

---

## 4. 测试覆盖率

| 文件 | 语句覆盖率 | 缺失行数 | 缺失位置 |
|-----|-----------|---------|---------|
| app.py | 95% | 1 | 第27行（开发服务器启动代码） |
| mcdonald_predictor.py | 100% | 0 | - |
| test_api.py | 100% | 0 | - |
| test_integration.py | 97% | 2 | 第95、97行（测试数据验证） |
| test_mcdonald_predictor.py | 100% | 0 | - |

**整体覆盖率：99%**

---

## 5. 发现的Bug及修复

### Bug 1：visit_frequency列类型错误
- **问题**：代码尝试将字符串类型的visit_frequency与数字0比较
- **修复**：移除了visit_frequency > 0的条件，改为仅过滤年龄和缺失值

### Bug 2：目标列名错误
- **问题**：代码中使用'likes_mcdonald'，但数据文件中是'liked_mcdonalds'
- **修复**：修正了列名引用

### Bug 3：缺少特征编码
- **问题**：未对分类变量（gender, visit_frequency, satisfaction_level）进行编码
- **修复**：添加了pd.get_dummies进行独热编码

### Bug 4：缺少输入验证
- **问题**：缺少字段会导致KeyError
- **修复**：添加了必填字段验证和API错误处理

---

## 6. 测试结论

✅ **所有测试通过** ✅

- 共执行21个测试用例，全部通过
- 整体测试覆盖率99%
- 所有发现的Bug均已修复
- 系统功能完整，API接口稳定可靠
- 错误处理机制完善

系统可以正常部署和使用。
