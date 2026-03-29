/**
 * 客户路由 - 路由层
 * 负责API路由定义
 */

const express = require('express');
const router = express.Router();
const customerController = require('../controllers/customerController');

/**
 * @route POST /api/customers/predict
 * @desc 预测客户生命周期价值
 * @access Public
 */
router.post('/predict', customerController.predictCLV);

/**
 * @route GET /api/customers
 * @desc 获取所有客户记录
 * @access Public
 */
router.get('/', customerController.getAllCustomers);

/**
 * @route GET /api/customers/statistics
 * @desc 获取统计数据
 * @access Public
 */
router.get('/statistics', customerController.getStatistics);

/**
 * @route GET /api/customers/:id
 * @desc 根据ID获取客户记录
 * @access Public
 */
router.get('/:id', customerController.getCustomerById);

/**
 * @route DELETE /api/customers/:id
 * @desc 删除客户记录
 * @access Public
 */
router.delete('/:id', customerController.deleteCustomer);

/**
 * @route DELETE /api/customers
 * @desc 清空所有客户记录
 * @access Public
 */
router.delete('/', customerController.clearAllCustomers);

module.exports = router;
