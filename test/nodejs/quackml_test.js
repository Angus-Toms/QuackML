var duckdb = require('../../duckdb/tools/nodejs');
var assert = require('assert');

describe(`quackml extension`, () => {
    let db;
    let conn;
    before((done) => {
        db = new duckdb.Database(':memory:', {"allow_unsigned_extensions":"true"});
        conn = new duckdb.Connection(db);
        conn.exec(`LOAD '${process.env.QUACKML_EXTENSION_BINARY_PATH}';`, function (err) {
            if (err) throw err;
            done();
        });
    });

    it('quackml function should return expected string', function (done) {
        db.all("SELECT quackml('Sam') as value;", function (err, res) {
            if (err) throw err;
            assert.deepEqual(res, [{value: "Quackml Sam 🐥"}]);
            done();
        });
    });

    it('quackml_openssl_version function should return expected string', function (done) {
        db.all("SELECT quackml_openssl_version('Michael') as value;", function (err, res) {
            if (err) throw err;
            assert(res[0].value.startsWith('Quackml Michael, my linked OpenSSL version is OpenSSL'));
            done();
        });
    });
});