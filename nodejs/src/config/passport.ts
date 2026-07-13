import bcrypt from "bcryptjs";
import passport from "passport";
import { Strategy as LocalStrategy } from "passport-local";
import { findUserByUsername } from "~/db/users.js";

passport.use(
  new LocalStrategy(async function (username, password, done) {
    let user;
    try {
      user = await findUserByUsername(username);
    } catch (err) {
      return done(null, false, { message: "Incorrect username or password." });
    }
    if (!user) {
      return done(null, false, { message: "Incorrect username or password." });
    }
    if (await bcrypt.compare(password, user.password)) {
      return done(null, user);
    }
    return done(null, false, { message: "Incorrect username or password." });
  }),
);
passport.serializeUser(function (user, done) {
  done(null, { id: user.id });
});
passport.deserializeUser(function (user: { id: string }, done) {
  done(null, user);
});
export default passport;
