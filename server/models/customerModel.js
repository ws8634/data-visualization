/**
 * 客户数据模型 - 数据层
 * 负责数据的存储和获取
 */

const fs = require('fs');
const path = require('path');

const DATA_FILE = path.join(__dirname, '../../data/customers.json');

/**
 * 确保数据文件和目录存在
 */
function ensureDataFile() {
  const dir = path.dirname(DATA_FILE);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  if (!fs.existsSync(DATA_FILE)) {
    fs.writeFileSync(DATA_FILE, JSON.stringify([]), 'utf8');
  }
}

/**
 * 获取所有客户记录
 * @returns {Array} 客户记录数组
 */
function getAllCustomers() {
  ensureDataFile();
  const data = fs.readFileSync(DATA_FILE, 'utf8');
  return JSON.parse(data);
}

/**
 * 添加新的客户记录
 * @param {Object} customer - 客户数据
 * @returns {Object} 添加后的客户记录（包含id和时间戳）
 */
function addCustomer(customer) {
  ensureDataFile();
  const customers = getAllCustomers();
  const newCustomer = {
    ...customer,
    id: Date.now(),
    createdAt: new Date().toISOString()
  };
  customers.unshift(newCustomer);
  fs.writeFileSync(DATA_FILE, JSON.stringify(customers, null, 2), 'utf8');
  return newCustomer;
}

/**
 * 根据ID获取客户记录
 * @param {number} id - 客户ID
 * @returns {Object|null} 客户记录，不存在返回null
 */
function getCustomerById(id) {
  const customers = getAllCustomers();
  return customers.find(c => c.id === id) || null;
}

/**
 * 删除客户记录
 * @param {number} id - 客户ID
 * @returns {boolean} 是否删除成功
 */
function deleteCustomer(id) {
  const customers = getAllCustomers();
  const index = customers.findIndex(c => c.id === id);
  if (index === -1) return false;
  customers.splice(index, 1);
  fs.writeFileSync(DATA_FILE, JSON.stringify(customers, null, 2), 'utf8');
  return true;
}

/**
 * 清空所有客户记录
 * @returns {boolean} 是否清空成功
 */
function clearAllCustomers() {
  fs.writeFileSync(DATA_FILE, JSON.stringify([]), 'utf8');
  return true;
}

module.exports = {
  getAllCustomers,
  addCustomer,
  getCustomerById,
  deleteCustomer,
  clearAllCustomers
};
