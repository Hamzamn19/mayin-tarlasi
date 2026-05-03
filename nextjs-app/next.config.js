/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/py/detect',
        destination: 'http://localhost:8000/detect',
      },
    ]
  },
}

module.exports = nextConfig
