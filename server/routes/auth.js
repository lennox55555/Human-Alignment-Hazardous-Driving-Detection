const express = require('express');
const { createToken } = require('../utils/token');
const router = express.Router();
const User = require('../models/user');

router.post('/register', async (req, res) => {
    const { email, password, ...formData } = req.body;
    try {
        const user = await User.register(email, password, formData)
        const token = createToken(user._id)

        res.status(200).json({ email, token })
    } catch (err) {
        res.status(400).json({ err: err.message })
    }
});

router.post('/signIn', async (req, res) => {
    const { email, password } = req.body;
    console.log(email, password)
    try {
        const user = await User.signIn(email, password)
        const token = createToken(user._id)

        res.status(200).json({ email, token })
    } catch (err) {
        res.status(400).json({ err: err.message })
    }
});

module.exports = router;