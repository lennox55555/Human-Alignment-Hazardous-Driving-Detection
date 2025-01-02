export type RegistrationFormData = {
    email: string,
    password: string,
    state: string,
    city: string,
    licenseAge: string,
    age: number,
    ethnicity: string,
    carMakeModel: string,
    gender: string,
    speedingTicket: boolean,
    dui: boolean,
    visuallyImpaired: boolean
}

export type SignInFormData = {
    email: string,
    password: string
}