const fs = require("fs");
const local = require("./localmod");
import("./dyn").then(m => m);
module.exports = { alpha, beta: function () {} };
exports.gamma = () => 1;
function alpha() {}
class K extends local.Base { method() {} static s() {} }
export default function () {}
