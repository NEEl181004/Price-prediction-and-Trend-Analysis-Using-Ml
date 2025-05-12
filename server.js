const express = require('express');
const { Pool } = require('pg');
const cors = require('cors');
const app = express();
const port = 3001;

app.use(cors());

const pool = new Pool({
  user: 'miniproject',
  host: 'localhost',
  database: 'price_comparison',
  password: '31998369',
  port: 6543,
});

app.get('/api/price-comparison', async (req, res) => {
  const query = req.query.query;
  if (!query) {
    return res.status(400).json({ error: 'Query parameter is required' });
  }

  try {
    const result = await pool.query(`
      SELECT DISTINCT ON (platform) 
        product_name, platform, price, rating, num_reviews, timestamp
      FROM product_prices4
      WHERE LOWER(product_name) LIKE $1
      ORDER BY platform, timestamp DESC
    `, [`%${query.toLowerCase()}%`]);

    res.json(result.rows);
  } catch (err) {
    console.error('DB Error:', err);
    res.status(500).json({ error: 'Failed to fetch latest product data' });
  }
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
