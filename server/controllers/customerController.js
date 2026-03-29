/**
 * 客户控制器 - 接口层
 * 负责处理HTTP请求和响应
 */

const customerModel = require('../models/customerModel');
const customerService = require('../services/customerService');

/**
 * 预测客户价值
 * @param {Object} req - Express请求对象
 * @param {Object} res - Express响应对象
 */
function predictCLV(req, res) {
  try {
    const data = req.body;
    
    const validation = customerService.validateCustomerData(data);
    if (!validation.isValid) {
      return res.status(400).json({
        success: false,
        message: '数据验证失败',
        errors: validation.errors
      });
    }

    const clv = customerService.calculateCLV(data);
    const level = customerService.getCLVLevel(clv);

    const record = {
      ...data,
      clv,
      level: level.label
    };

    const savedRecord = customerModel.addCustomer(record);

    res.json({
      success: true,
      data: {
        ...savedRecord,
        levelInfo: level
      }
    });
  } catch (error) {
    console.error('预测CLV失败:', error);
    res.status(500).json({
      success: false,
      message: '服务器内部错误',
      error: error.message
    });
  }
}

/**
 * 获取所有客户记录
 * @param {Object} req - Express请求对象
 * @param {Object} res - Express响应对象
 */
function getAllCustomers(req, res) {
  try {
    const customers = customerModel.getAllCustomers();
    res.json({
      success: true,
      data: customers
    });
  } catch (error) {
    console.error('获取客户记录失败:', error);
    res.status(500).json({
      success: false,
      message: '服务器内部错误',
      error: error.message
    });
  }
}

/**
 * 根据ID获取客户记录
 * @param {Object} req - Express请求对象
 * @param {Object} res - Express响应对象
 */
function getCustomerById(req, res) {
  try {
    const id = parseInt(req.params.id);
    const customer = customerModel.getCustomerById(id);
    
    if (!customer) {
      return res.status(404).json({
        success: false,
        message: '客户记录不存在'
      });
    }

    res.json({
      success: true,
      data: customer
    });
  } catch (error) {
    console.error('获取客户记录失败:', error);
    res.status(500).json({
      success: false,
      message: '服务器内部错误',
      error: error.message
    });
  }
}

/**
 * 删除客户记录
 * @param {Object} req - Express请求对象
 * @param {Object} res - Express响应对象
 */
function deleteCustomer(req, res) {
  try {
    const id = parseInt(req.params.id);
    const success = customerModel.deleteCustomer(id);
    
    if (!success) {
      return res.status(404).json({
        success: false,
        message: '客户记录不存在'
      });
    }

    res.json({
      success: true,
      message: '删除成功'
    });
  } catch (error) {
    console.error('删除客户记录失败:', error);
    res.status(500).json({
      success: false,
      message: '服务器内部错误',
      error: error.message
    });
  }
}

/**
 * 获取统计数据
 * @param {Object} req - Express请求对象
 * @param {Object} res - Express响应对象
 */
function getStatistics(req, res) {
  try {
    const customers = customerModel.getAllCustomers();
    const stats = customerService.getStatistics(customers);

    res.json({
      success: true,
      data: stats
    });
  } catch (error) {
    console.error('获取统计数据失败:', error);
    res.status(500).json({
      success: false,
      message: '服务器内部错误',
      error: error.message
    });
  }
}

/**
 * 清空所有客户记录
 * @param {Object} req - Express请求对象
 * @param {Object} res - Express响应对象
 */
function clearAllCustomers(req, res) {
  try {
    customerModel.clearAllCustomers();
    res.json({
      success: true,
      message: '清空成功'
    });
  } catch (error) {
    console.error('清空客户记录失败:', error);
    res.status(500).json({
      success: false,
      message: '服务器内部错误',
      error: error.message
    });
  }
}

module.exports = {
  predictCLV,
  getAllCustomers,
  getCustomerById,
  deleteCustomer,
  getStatistics,
  clearAllCustomers
};
