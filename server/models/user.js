const mongoose = require('mongoose');
const bcrypt = require('bcrypt');

const formSchema = new mongoose.Schema({
    state: { type: String, required: true },
    city: { type: String, required: true },
    licenseAge: { type: String, required: true },
    age: { type: Number, required: true, min: 0 },
    ethnicity: { type: String, required: false },
    carMakeModel: { type: String, required: true },
    gender: { type: String, required: true },
    speedingTicket: { type: Boolean, default: false },
    dui: { type: Boolean, default: false },
    visuallyImpaired: { type: Boolean, default: false },
});

const userSchema = new mongoose.Schema({
    email: { type: String, required: true, unique: true},
    password: { type: String, required: true },
    form: { type: formSchema, required: true }
});

userSchema.statics.register = async function (email, password, formData) {
    const emailExists = await this.findOne({ email })
 
    if (emailExists) {
        throw Error('Email already exists!')
    }

    const salt = await bcrypt.genSalt(10);
    const hash = await bcrypt.hash(password, salt);

    const user = await this.create({ 
        email, 
        password: hash, 
        form: formData 
    });

    return user;
};

userSchema.statics.signIn = async function (email, password) {
    const user = await this.findOne({ email })

    if (!user) {
        throw Error('The email that was provided is not valid, please try again');
    }

    const match = await bcrypt.compare(password, user.password);

    if (!match) {
        throw Error('Incorrect password');
    }

    return user;
}

const User = mongoose.model('User', userSchema)

module.exports = User;