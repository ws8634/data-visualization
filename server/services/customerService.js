/**
 * 客户服务 - 业务逻辑层
 * 负责CLV计算和数据处理
 */

/**
 * 计算客户生命周期价值（CLV）
 * @param {Object} customer - 客户信息
 * @param {string} customer.gender - 性别
 * @param {number} customer.age - 年龄
 * @param {number} customer.income - 月收入
 * @param {number} customer.purchaseFrequency - 购买频率（次/月）
 * @param {number} customer.avgPurchaseAmount - 平均消费金额
 * @param {number} customer.satisfaction - 满意度评分（1-10）
 * @param {number} customer.customerYears - 客户年限（年）
 * @param {string} customer.industry - 所属行业
 * @returns {number} CLV值
 */
function calculateCLV(customer) {
  const baseCLV = customer.avgPurchaseAmount * customer.purchaseFrequency * 12;
  const loyaltyFactor = 1 + (customer.customerYears * 0.1);
  const satisfactionFactor = customer.satisfaction / 5;
  const incomeFactor = Math.min(customer.income / 10000, 2);
  
  let industryFactor = 1;
  switch (customer.industry) {
    case 'finance': industryFactor = 1.3; break;
    case 'technology': industryFactor = 1.25; break;
    case 'healthcare': industryFactor = 1.2; break;
    case 'retail': industryFactor = 0.9; break;
    default: industryFactor = 1;
  }

  let genderFactor = 1;
  switch (customer.gender) {
    case 'female': genderFactor = 1.1; break;
    case 'male': genderFactor = 1.0; break;
    default: genderFactor = 1.05;
  }

  const ageFactor = customer.age < 30 ? 1.2 : (customer.age > 50 ? 0.8 : 1.0);

  return Math.round(baseCLV * loyaltyFactor * satisfactionFactor * incomeFactor * industryFactor * genderFactor * ageFactor);
}

/**
 * 获取CLV等级
 * @param {number} clv - CLV值
 * @returns {Object} 等级信息
 */
function getCLVLevel(clv) {
  if (clv >= 100000) {
    return { label: '高价值', class: 'badge-high' };
  } else if (clv >= 30000) {
    return { label: '中价值', class: 'badge-medium' };
  } else {
    return { label: '低价值', class: 'badge-low' };
  }
}

/**
 * 验证客户数据
 * @param {Object} data - 要验证的数据
 * @returns {Object} 验证结果
 */
function validateCustomerData(data) {
  const errors = {};
  let isValid = true;

  if (!data.name || data.name.trim().length < 2) {
    errors.name = '请输入有效的客户姓名（至少2个字符）';
    isValid = false;
  }

  if (!data.age || data.age < 18 || data.age > 100) {
    errors.age = '请输入18-100之间的有效年龄';
    isValid = false;
  }

  if (!data.gender) {
    errors.gender = '请选择性别';
    isValid = false;
  }

  if (!data.income || data.income < 0) {
    errors.income = '请输入有效的月收入';
    isValid = false;
  }

  if (!data.purchaseFrequency || data.purchaseFrequency < 0) {
    errors.purchaseFrequency = '请输入有效的购买频率';
    isValid = false;
  }

  if (!data.avgPurchaseAmount || data.avgPurchaseAmount < 0) {
    errors.avgPurchaseAmount = '请输入有效的平均消费金额';
    isValid = false;
  }

  if (!data.satisfaction || data.satisfaction < 1 || data.satisfaction > 10) {
    errors.satisfaction = '请输入1-10之间的满意度评分';
    isValid = false;
  }

  if (!data.customerYears || data.customerYears < 0) {
    errors.customerYears = '请输入有效的客户年限';
    isValid = false;
  }

  if (!data.industry) {
    errors.industry = '请选择所属行业';
    isValid = false;
  }

  return { isValid, errors };
}

/**
 * 获取统计数据
 * @param {Array} customers - 客户记录数组
 * @returns {Object} 统计数据
 */
function getStatistics(customers) {
  if (customers.length === 0) return null;

  const totalCLV = customers.reduce((sum, record) => sum + record.clv, 0);
  const avgCLV = Math.round(totalCLV / customers.length);
  const maxCLV = Math.max(...customers.map(r => r.clv));
  const minCLV = Math.min(...customers.map(r => r.clv));

  const industryStats = {};
  customers.forEach(record => {
    if (!industryStats[record.industry]) {
      industryStats[record.industry] = { total: 0, count: 0 };
    }
    industryStats[record.industry].total += record.clv;
    industryStats[record.industry].count++;
  });

  return {
    totalCLV,
    avgCLV,
    maxCLV,
    minCLV,
    count: customers.length,
    industryStats
  };
}

module.exports = {
  calculateCLV,
  getCLVLevel,
  validateCustomerData,
  getStatistics
};
