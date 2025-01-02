const jwt = require('jsonwebtoken');

const createToken = (_id) => {
    return jwt.sign({ _id}, process.env.SECRET, { expiresIn: '2d'})
}

module.exports = { createToken }