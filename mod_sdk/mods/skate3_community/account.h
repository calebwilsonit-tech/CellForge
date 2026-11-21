#pragma once
// Saved sign-in (encrypted for the current Windows user) and the small in-game sign-in / create-account windows.

#include <string>

namespace s3c {

struct Credentials {
  std::string npid;
  std::string password;
  std::string token;  // RPCN email token (needed on every login for accounts with email verification)
};

bool load_credentials(Credentials& out);
void save_credentials(const Credentials& c);
void forget_credentials();

struct SignInForm {
  std::string npid, password, token, email;
  std::string message;  // shown at the top of the window (why we are asking / what went wrong)
};

enum class SignInAction { SignIn, CreateAccount, NotNow };

// Both block until the player answers; they run their own message loop (call from a worker thread).
SignInAction prompt_signin(SignInForm& form);
bool prompt_create_account(SignInForm& form);  // true = "Create" pressed

}  // namespace s3c
