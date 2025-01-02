const jwt = require('jsonwebtoken');
const User = require('../models/user');

// Include requireAuth middleware for any Survey Routes (e.g. router.use(requireAuth))

const requireAuth = async (req, res, next) => {

    const { authorization } = res.headers;

    if (!authorization) {
        return res.status(401).json({error: 'Authorization token is required!'})
    }
        const token = authorization.split(' ')[1]
        try {
            const { _id } = jwt.verify(token, process.env.SECRET)
            
            req.user = await User.findOne({_id}).select('_id');
            next();

        } catch (err) {
            res.status(401).json({error: 'Request is not authorized'})
        }
}

module.exports = requireAuth